import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

dataset = [Data(x=torch.randn(10, 5), y=torch.randn(10, 2)) for _ in range(4)]
loader = DataLoader(dataset, batch_size=2)
batch = next(iter(loader))
print("batch.num_graphs:", getattr(batch, 'num_graphs', None))
print("batch.batch:", getattr(batch, 'batch', None))
