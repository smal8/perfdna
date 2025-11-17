import os
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd
import networkx as nx # python package for creating, manipulating, and studying graph networks

data_dir = Path.cwd() / "data" / "cora"
data_dir.mkdir(parents=True, exist_ok=True)

url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
tgz_path = data_dir / "cora.tgz"

if not tgz_path.exists():
    print("Downloading cora.tgz...")
    with urllib.request.urlopen(url) as response, open(tgz_path, "wb") as out_file:
        out_file.write(response.read())
    print("Done.")

# extract dataset from tgz
extracted_marker = data_dir / "cora.cites"
if not extracted_marker.exists():
    print("Extracting cora.tgz")
    # extracts into cora.cites cora.content
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir.parent)
    print("Done extracting.")

edgelist = pd.read_csv(
    data_dir / "cora.cites",
    sep = "\t", # file is tab separated
    header = None,
    names = ["target", "source"] # column names
)

print(edgelist.sample(frac=1).head(5))