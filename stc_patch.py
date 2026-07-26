import re

with open("physics_gat_pipeline.py", "r") as f:
    content = f.read()

# We will apply edits directly to the string and rewrite.

# 1. Add STC_DIR
content = content.replace("OUT.mkdir(parents=True, exist_ok=True)",
"""OUT.mkdir(parents=True, exist_ok=True)
STC_DIR = OUT / "stc"
STC_DIR.mkdir(parents=True, exist_ok=True)""")

# 2. Add STC saving helper function to avoid repeating logic
# Find a good place to insert it (after extract_simulation)
helper = """
def save_stc(data_array: np.ndarray, evoked: mne.Evoked, algorithm: str, condition: str):
    vertices = [fwd_fixed["src"][0]["vertno"], fwd_fixed["src"][1]["vertno"]]
    tstep = evoked.times[1] - evoked.times[0] if len(evoked.times) > 1 else 1.0 / CFG.sfreq
    stc = mne.SourceEstimate(
        data=data_array,
        vertices=vertices,
        tmin=evoked.times[0],
        tstep=tstep,
        subject="sample"
    )
    stc.save(str(STC_DIR / f"{algorithm}_{condition}"), overwrite=True)

"""
content = content.replace("class SourceDataset", helper + "class SourceDataset")

# 3. Add save_stc calls into the real EEG inference loop
# Find tikhonov/sparse loop
tikh_loop_search = """                predictions[algorithm] = (
                    output.abs().amax(-1)[0].cpu().numpy() * target_scale,
                    preprocess_ms + (time.perf_counter() - started) * 1000.0,
                )"""
tikh_loop_replace = """                raw_data = output[0].cpu().numpy() * target_scale
                save_stc(raw_data, evoked, algorithm, condition)
                predictions[algorithm] = (
                    np.max(np.abs(raw_data), axis=1),
                    preprocess_ms + (time.perf_counter() - started) * 1000.0,
                )"""
content = content.replace(tikh_loop_search, tikh_loop_replace)

# Find GAT loop
gat_loop_search = """            predictions["physics_gat"] = (
                output.abs().amax(-1)[0].cpu().numpy() * target_scale,
                preprocess_ms + (time.perf_counter() - started) * 1000.0,
            )"""
gat_loop_replace = """            gat_raw_data = output[0].cpu().numpy() * target_scale
            save_stc(gat_raw_data, evoked, "physics_gat", condition)
            predictions["physics_gat"] = (
                np.max(np.abs(gat_raw_data), axis=1),
                preprocess_ms + (time.perf_counter() - started) * 1000.0,
            )"""
content = content.replace(gat_loop_search, gat_loop_replace)

# Find convdip loop
convdip_search = """                convdip_prediction = convdip_model.predict(evoked)
                convdip_ms = (time.perf_counter() - started) * 1000.0
                convdip_map = np.max(np.abs(convdip_to_array(convdip_prediction)[0]), axis=1)"""
convdip_replace = """                convdip_prediction = convdip_model.predict(evoked)
                convdip_ms = (time.perf_counter() - started) * 1000.0
                cd_array = convdip_to_array(convdip_prediction)[0]
                save_stc(cd_array, evoked, "convdip", condition)
                convdip_map = np.max(np.abs(cd_array), axis=1)"""
content = content.replace(convdip_search, convdip_replace)


with open("physics_gat_pipeline.py", "w") as f:
    f.write(content)

print("Patch applied.")
