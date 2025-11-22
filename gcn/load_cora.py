from torch_geometric.datasets import Planetoid
from pathlib import Path

def load_cora():
    root = Path("data/cora")
    dataset = Planetoid(root=str(root), name="Cora")
    data = dataset[0]
    return data

if __name__ == "__main__":
    data = load_cora()
    print(data)


"""
CORA is a citation network.
Nodes - papers (2708)
Edges - citations(10556)
Node features - bag-of-words-vectors (1433 dimensional binary vectors)
Labels - subject category (7 classes)

Each paper has a 1433-dimensional sparse word vector (x)
A category label (y)
connections to other papers it cites

cora.content
 - each row is <PaperId> <1433 features> <Label>

cora.cites
 - each row is <PaperID_1> <PaperID_2>

data.x is [2708, 1433] PyG tensor
data.edge_index is [2, 10556] 
 - 10556 edges, using 2 rows: source_nodes, target_nodes
 - to optimize for sparse graphs
data.y is the class label for each node

CORA is a semi-supervised benchmark
 - train_mask are nodes used for training
 - val_mask are validation nodes
 - test_mask are held out eval nodes

dataset[0] is a PyG data object, these are single-graph datasets
There is one graph containing all nodes and edges (i.e. dataset[1] doesn't exist)
Train/val/test splits happen via the masks

"""