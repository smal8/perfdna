import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from gcn import GCN

def main():

    # load dataset
    dataset = Planetoid(root = "data/cora", name = "Cora")
    data = dataset[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = data.to(device)

    model = GCN(
        in_dim = dataset.num_node_features,
        hid_dim = 15,
        out_dim = dataset.num_classes
    ).to(device)

    # only eager mode inference
    model.eval()

    with torch.no_grad():
        start = time.time()
        out = model(data.x, data.edge_index)
        # to ensure all CUDA kernels complete before coming back to CPU, 
        # accurately measures CUDA kernel execution time
        # because GPU ops are async, the CPU queues ops to the GPU
        # and continues executing subsequent python code
        torch.cuda.synchronize()
        end = time.time()
    
    print(f"Eager inference runtime: {end - start:.6f} seconds")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    main()
    