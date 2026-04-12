import json
with open("notebooks/03_classification.ipynb", "r") as f:
    nb = json.load(f)
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        print(f"=== CELL {idx+1} ===")
        print("".join(cell["source"]))
