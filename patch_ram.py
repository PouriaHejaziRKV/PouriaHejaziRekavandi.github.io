import re

with open("physics_gat_pipeline.py", "r") as f:
    content = f.read()

# Modify SourceDataset to calculate Tikhonov on the fly to save 50GB of RAM
new_dataset_class = """class SourceDataset(Dataset):
    def __init__(self, eeg: np.ndarray, target: np.ndarray,
                 graph: bool, input_scale: float):
        self.eeg = eeg
        self.target = target
        self.graph = graph
        self.input_scale = input_scale
        self.lead_field = LEAD_FIELD.T

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, index: int) -> Data:
        # Calculate Tikhonov on the fly to save RAM (40000 samples * 5120 vertices takes 50GB if pre-computed)
        eeg_sample = self.eeg[index]
        initial_raw = TIKHONOV_OPERATOR @ eeg_sample
        initial = np.clip(initial_raw / self.input_scale, -10, 10).astype(np.float32)

        edge_index, edge_attr = (GRAPH_BUILDER(initial)
                                 if self.graph else (None, None))

        node_features = np.concatenate([initial, self.lead_field], axis=1)

        return Data(x=torch.from_numpy(node_features),
                    y=torch.from_numpy(self.target[index]),
                    eeg=torch.from_numpy(eeg_sample),
                    edge_index=edge_index, edge_attr=edge_attr,
                    pos=torch.from_numpy(COORDINATES_MM))"""

# We need to replace the old SourceDataset
old_dataset_search = r"class SourceDataset\(Dataset\):.*?pos=torch\.from_numpy\(COORDINATES_MM\)\)\s+# Added pos for unpooling"
content = re.sub(old_dataset_search, new_dataset_class, content, flags=re.DOTALL)

# Now we need to fix the data loading loops that were pre-computing train_initial
loop_search = r"""    train_initial_raw = np\.einsum\("vc,bct->bvt", TIKHONOV_OPERATOR,
                                  train_eeg, optimize=True\)\.astype\(np\.float32\)
.*?
    test_target = \{name: normalize_target\(source\) for name, \(_, source\) in test_raw\.items\(\)\}"""

loop_replace = """    # Calculate scales using a smaller subset to avoid 50GB RAM crash on full dataset
    subset_size = min(len(train_eeg), 2000)
    subset_eeg = train_eeg[:subset_size]
    subset_initial_raw = np.einsum("vc,bct->bvt", TIKHONOV_OPERATOR, subset_eeg, optimize=True).astype(np.float32)

    input_scale = max(float(np.percentile(np.abs(subset_initial_raw), 99.5)), 1e-12)
    target_scale = max(float(np.percentile(np.abs(train_target_raw), 99.5)), 1e-12)

    normalize_target = lambda x: np.clip(x / target_scale, -10, 10).astype(np.float32)

    train_target = normalize_target(train_target_raw)
    validation_target = normalize_target(validation_target_raw)
    test_target = {name: normalize_target(source) for name, (_, source) in test_raw.items()}
"""
content = re.sub(loop_search, loop_replace, content, flags=re.DOTALL)

# Fix PyGDataLoader calls to match new SourceDataset signature
content = content.replace("""graph_train = PyGDataLoader(SourceDataset(train_initial, train_target,
                                               train_eeg, True),""",
"""graph_train = PyGDataLoader(SourceDataset(train_eeg, train_target,
                                               True, input_scale),""")
content = content.replace("""graph_validation = PyGDataLoader(SourceDataset(validation_initial,
                                                    validation_target,
                                                    validation_eeg, True),""",
"""graph_validation = PyGDataLoader(SourceDataset(validation_eeg,
                                                    validation_target,
                                                    True, input_scale),""")
content = content.replace("""plain_train = PyGDataLoader(SourceDataset(train_initial, train_target,
                                               train_eeg, False),""",
"""plain_train = PyGDataLoader(SourceDataset(train_eeg, train_target,
                                               False, input_scale),""")
content = content.replace("""plain_validation = PyGDataLoader(SourceDataset(validation_initial,
                                                    validation_target,
                                                    validation_eeg, False),""",
"""plain_validation = PyGDataLoader(SourceDataset(validation_eeg,
                                                    validation_target,
                                                    False, input_scale),""")

# Fix the test loop that uses test_initial
test_search = r"""    for condition, \(physical_eeg, target_raw\) in test_raw\.items\(\):
        initial = test_initial\[condition\]"""
test_replace = """    for condition, (physical_eeg, target_raw) in test_raw.items():
        initial_raw = np.einsum("vc,bct->bvt", TIKHONOV_OPERATOR, physical_eeg, optimize=True).astype(np.float32)
        initial = np.clip(initial_raw / input_scale, -10, 10).astype(np.float32)"""
content = re.sub(test_search, test_replace, content)

test_loader_search = r"""loader = PyGDataLoader\(SourceDataset\(initial, target_normalized,
                                                  physical_eeg, True\),"""
test_loader_replace = """loader = PyGDataLoader(SourceDataset(physical_eeg, target_normalized,
                                                  True, input_scale),"""
content = re.sub(test_loader_search, test_loader_replace, content)


with open("physics_gat_pipeline.py", "w") as f:
    f.write(content)

print("Patch applied.")
