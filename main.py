import os
import gc
import subprocess
import sys
import logging
import random
import shutil
import joblib
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Install dependencies if not already installed
try:
    import mne
    import esinet
    from esinet import Simulation, Net
    from esinet.forward import create_forward_model, get_info
except ImportError:
    print("Installing/Updating dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "mne", "esinet", "scipy", "scikit-learn"])
    import mne
    import esinet
    from esinet import Simulation, Net
    from esinet.forward import create_forward_model, get_info

from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def to_array(x):
    if hasattr(x, "data"): return np.asarray(x.data)
    x = np.asarray(x)
    return x if x.ndim == 2 else x[:, None]

def multisource_metrics(gt, pr, pos_mm, adj):
    def get_local_maxima(vec):
        vec = vec.ravel()
        if vec.size == 0:
            return np.array([], dtype=int)

        if np.all(vec <= 0):
            return np.array([np.argmax(vec)])

        # Vectorized local maxima detection
        neighbor_indices = adj.indices
        row_ptr = adj.indptr
        node_indices = np.repeat(np.arange(len(vec)), np.diff(row_ptr))

        # A node is NOT a local maximum if any neighbor is > to it
        # (Using >= might be too strict if neighbors have equal values)
        not_max_mask = vec[neighbor_indices] > vec[node_indices]
        is_not_max = np.zeros(len(vec), dtype=bool)
        is_not_max[node_indices[not_max_mask]] = True

        has_neighbors = np.diff(row_ptr) > 0
        cand_mask = (~is_not_max) & (vec > 0) & has_neighbors
        cand = np.where(cand_mask)[0]

        if cand.size == 0:
            return np.array([np.argmax(vec)])

        # Sort candidates by value descending
        cand = cand[np.argsort(vec[cand])[::-1]]

        selected = []
        for i in cand:
            if not selected:
                selected.append(i)
            else:
                # Spatial pruning: keep only maxima at least 30mm apart
                d = np.linalg.norm(pos_mm[selected] - pos_mm[i], axis=1)
                if np.all(d > 30.0):
                    selected.append(i)
        return np.array(selected)

    gt_max = get_local_maxima(gt)
    pr_max = get_local_maxima(pr)

    if gt_max.size == 0 or pr_max.size == 0:
        return 100.0, 0.0

    try:
        D = cdist(pos_mm[gt_max], pos_mm[pr_max])
        if D.size == 0: return 100.0, 0.0
        min_d = D.min(axis=1)
        return np.mean(min_d), np.mean(min_d <= 30.0) * 100
    except Exception as e:
        logging.error(f"Distance calculation error: {e}")
        return 100.0, 0.0

def median_mad(data):
    arr = np.array(data)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0: return np.nan, np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return med, mad

def initialize_pipeline():
    logging.info("Initializing Forward Models...")
    info = get_info(sfreq=100)
    fwd = create_forward_model(info=info, sampling='ico4')

    # Setup for testing (avoiding Inverse Crime)
    info_test = info.copy()
    for i in range(len(info_test['chs'])):
        info_test['chs'][i]['loc'][:3] += np.random.normal(0, 0.002, 3)
    fwd_test = create_forward_model(info=info_test, sampling='ico4')

    sim_settings = {
        'method': 'standard', 'number_of_sources': (1, 6), 'extents': (21, 58),
        'amplitudes': (5, 10), 'shapes': 'gaussian', 'duration_of_trial': 1.0,
        'target_snr': (4.5, 4.5), 'beta_noise': (0, 0), 'source_spread': 'region_growing',
    }

    fwd_gm = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=True)
    pos_gm = np.vstack([s['rr'][s['vertno']] for s in fwd_gm['src']]) * 1000
    adj = mne.spatial_src_adjacency(fwd_gm['src']).tocsr()

    return info, fwd, info_test, fwd_test, sim_settings, pos_gm, adj

class GeneticOptimizer:
    def __init__(self, fwd, info, pos_gm, adj, sim_settings, population_size=6, generations=3):
        self.fwd = fwd
        self.info = info
        self.pos_gm = pos_gm
        self.adj = adj
        self.sim_settings = sim_settings
        self.pop_size = population_size
        self.generations = generations

    def fitness(self, params, train_sim, val_sim):
        epochs, lr, batch_size = params
        net = Net(self.fwd, model_type='convdip')

        try:
            # We try to pass learning_rate and batch_size.
            try:
                net.fit(train_sim, epochs=int(epochs), validation_split=0.1, learning_rate=lr, batch_size=int(batch_size))
            except TypeError:
                net.fit(train_sim, epochs=int(epochs), validation_split=0.1)

            y_true = val_sim.source_data
            y_pred = net.predict(val_sim)

            mle_list = []
            for i in range(len(y_true)):
                jt = to_array(y_true[i])[:, 0]
                jp = to_array(y_pred[i])[:, 0]

                if jt.size == 0 or jp.size == 0 or np.all(np.isnan(jp)):
                    mle_list.append(100.0)
                    continue

                mle, _ = multisource_metrics(np.abs(jt), np.abs(jp), self.pos_gm, self.adj)
                mle_list.append(mle)

            avg_mle = np.mean(mle_list)
            fitness_val = 1.0 / (avg_mle + 1e-6)
            return fitness_val, avg_mle
        except Exception as e:
            logging.error(f"Error during fitness evaluation: {e}")
            return 0, 1000
        finally:
            del net
            tf.keras.backend.clear_session()
            gc.collect()

    def optimize(self, train_sim, val_sim):
        logging.info("Starting GA optimization")

        # Initial Population: [epochs, learning_rate, batch_size]
        population = []
        for _ in range(self.pop_size):
            ind = [
                random.randint(50, 500),         # Epochs
                10**random.uniform(-4, -2),      # Learning Rate (1e-4 to 1e-2)
                random.choice([8, 16, 32, 64])   # Batch Size
            ]
            population.append(ind)

        best_ind = None
        best_fitness = -1
        best_mle = 1000

        for gen in range(self.generations):
            logging.info(f"GA Generation {gen+1}/{self.generations}")
            fitness_scores = []
            for ind in population:
                fit, mle = self.fitness(ind, train_sim, val_sim)
                fitness_scores.append(fit)
                if fit > best_fitness:
                    best_fitness, best_ind, best_mle = fit, ind, mle
                logging.info(f"  Params: Ep={int(ind[0])}, LR={ind[1]:.4f}, BS={int(ind[2])} -> MLE: {mle:.2f}")

            # Selection and Crossover
            new_population = []
            for _ in range(self.pop_size // 2):
                p1 = max(random.sample(list(zip(population, fitness_scores)), 2), key=lambda x: x[1])[0]
                p2 = max(random.sample(list(zip(population, fitness_scores)), 2), key=lambda x: x[1])[0]

                # Crossover
                c1 = [
                    int((p1[0] + p2[0]) / 2),
                    (p1[1] + p2[1]) / 2,
                    random.choice([p1[2], p2[2]])
                ]
                # Mutation
                if random.random() < 0.3: c1[0] = np.clip(c1[0] + random.randint(-50, 50), 1, 500)
                if random.random() < 0.3: c1[1] = np.clip(c1[1] * random.uniform(0.5, 2.0), 1e-5, 1e-1)

                new_population.append(c1)
                new_population.append(p1 if random.random() < 0.5 else p2)
            population = new_population[:self.pop_size]

        logging.info(f"GA Best: Ep={int(best_ind[0])}, LR={best_ind[1]:.4f}, BS={int(best_ind[2])} (MLE: {best_mle:.2f})")
        return best_ind

def get_persistent_simulation(fwd, info, settings, n_target, results_dir):
    sim_path = os.path.join(results_dir, 'simulation_data.joblib')
    if os.path.exists(sim_path):
        logging.info(f"Loading cached simulation from {sim_path}")
        sim_data = joblib.load(sim_path)
        # sim_data is expected to be a dict or a Simulation object
        # If it's a Simulation object, we check length
        n_current = len(sim_data.source_data)
        if n_current < n_target:
            n_needed = n_target - n_current
            logging.info(f"Simulating additional {n_needed} samples...")
            sim_new = Simulation(fwd, info, settings=settings)
            sim_new.simulate(n_samples=n_needed)

            # Use concatenate if they are numpy arrays
            sim_data.source_data = np.concatenate([sim_data.source_data, sim_new.source_data], axis=0)
            sim_data.eeg_data = np.concatenate([sim_data.eeg_data, sim_new.eeg_data], axis=0)
            joblib.dump(sim_data, sim_path)
        return sim_data
    else:
        logging.info(f"Creating new simulation with {n_target} samples")
        sim_all = Simulation(fwd, info, settings=settings)
        sim_all.simulate(n_samples=n_target)
        joblib.dump(sim_all, sim_path)
        return sim_all

def run_simulation():
    info, fwd, info_test, fwd_test, sim_settings, pos_gm, adj = initialize_pipeline()
    sample_sizes = [1000, 2000, 4000, 8000, 16000] # Example of increasing sizes
    results_dir = '/content/onedrive/Documents/EEG'
    if not os.path.exists('/content/onedrive'): results_dir = 'esinet_project_results'
    os.makedirs(results_dir, exist_ok=True)

    summary_file = os.path.join(results_dir, 'performance_report.txt')
    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as f: f.write("EEG Source Localization - Optimized Pipeline\n" + "="*40 + "\n")

    for n in sample_sizes:
        logging.info(f"\n{'='*20}\nPROCESSING SAMPLE SIZE: {n}\n{'='*20}")

        # Check if already done
        model_path = os.path.join(results_dir, f'trained_net_{n}.keras')
        if os.path.exists(model_path):
            logging.info(f"Sample size {n} already processed. Skipping.")
            continue

        # 1. Get Simulation Data (Reuse previous if possible)
        full_sim = get_persistent_simulation(fwd, info, sim_settings, n, results_dir)

        # Create a light copy for the current size
        current_sim = Simulation(fwd, info, settings=sim_settings)
        current_sim.source_data = full_sim.source_data[:n]
        current_sim.eeg_data = full_sim.eeg_data[:n]

        # 2. GA Optimization on a subset to find best hyperparams fast
        ga_train_n = min(n, 2000)
        ga_val_n = 200
        ga_train_sim = Simulation(fwd, info, settings=sim_settings)
        ga_train_sim.source_data = current_sim.source_data[:ga_train_n]
        ga_train_sim.eeg_data = current_sim.eeg_data[:ga_train_n]

        ga_val_sim = Simulation(fwd, info, settings=sim_settings)
        ga_val_sim.simulate(n_samples=ga_val_n)

        # Smaller population/generations for larger n to save time, but n is handled by GA using subset
        pop_size, gens = (6, 3)
        optimizer = GeneticOptimizer(fwd, info, pos_gm, adj, sim_settings, population_size=pop_size, generations=gens)
        best_params = optimizer.optimize(ga_train_sim, ga_val_sim)
        best_epochs, best_lr, best_bs = best_params

        # 3. Final Training with best parameters
        logging.info(f"Final Training (Ep={int(best_epochs)}, LR={best_lr:.4f}, BS={int(best_bs)})...")
        net = Net(fwd, model_type='convdip')
        # We manually set parameters if fit() doesn't support them.
        # This is a bit speculative without the source, but many Keras wrappers work this way.
        try:
            history = net.fit(current_sim, epochs=int(best_epochs), validation_split=0.1, learning_rate=best_lr, batch_size=int(best_bs))
        except TypeError:
            logging.warning("Net.fit does not support learning_rate or batch_size. Using default.")
            history = net.fit(current_sim, epochs=int(best_epochs), validation_split=0.1)

        # 4. Evaluation on independent test set
        logging.info("Evaluating on test set...")
        sim_test = Simulation(fwd_test, info_test, settings=sim_settings)
        sim_test.simulate(n_samples=500)
        y_true = sim_test.source_data
        y_pred = net.predict(sim_test)

        mle_l, auc_l, found_l = [], [], []
        for i in range(len(y_true)):
            jt, jp = to_array(y_true[i])[:, 0], to_array(y_pred[i])[:, 0]
            if jt.size == 0 or jp.size == 0: continue
            mle, found = multisource_metrics(np.abs(jt), np.abs(jp), pos_gm, adj)
            mle_l.append(mle); found_l.append(found)
            jt_b = (np.abs(jt) > 0).astype(int)
            if len(np.unique(jt_b)) > 1: auc_l.append(roc_auc_score(jt_b, np.abs(jp)) * 100)

        mle_m, mle_mad = median_mad(mle_l)
        auc_m, auc_mad = median_mad(auc_l)
        with open(summary_file, 'a') as f:
            f.write(f"\nSize: {n} | Ep: {int(best_epochs)} | LR: {best_lr:.4f} | BS: {int(best_bs)}\n")
            f.write(f"MLE: {mle_m:.2f} (±{mle_mad:.2f}) | AUC: {auc_m:.2f}% | Found: {np.mean(found_l):.2f}%\n")

        # Save results
        net.model.save(model_path)
        plt.figure(); plt.plot(history.history['loss'], label='Loss'); plt.title(f'Training Loss (n={n})'); plt.savefig(os.path.join(results_dir, f'training_{n}.png')); plt.close()

        # Memory Cleanup
        del net, current_sim, sim_test, ga_train_sim, ga_val_sim; gc.collect(); tf.keras.backend.clear_session()

if __name__ == "__main__":
    run_simulation()
