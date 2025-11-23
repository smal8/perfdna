import time
import torch
from torch_geometric.datasets import Planetoid
from gcn import GCN

def main():
    dataset = Planetoid(root = "data/cora", name = "Cora")
    data = dataset[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    model = GCN(
        in_dim = dataset.num_node_features,
        hid_dim = 16,
        out_dim = dataset.num_classes
    ).to(device)

    model.eval()

    compiled_model = torch.compile(model)

    with torch.no_grad():
        start = time.time()
        out = compiled_model(data.x, data.edge_index)
        torch.cuda.synchronize()
        end = time.time()
    
    print(f"Compiled inference runtime: {end - start: .6f} seconds")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    main()