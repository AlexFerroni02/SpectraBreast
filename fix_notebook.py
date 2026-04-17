import json

with open('/home/y222/y222446/SpectraBreast/notebooks/02_pretraining.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "mask_np = mask.transpose(1, 2).cpu().numpy()" in line:
                source[i] = "    import torch.nn.functional as F\n"
                source.insert(i+1, "    mask_t = mask.transpose(1, 2)\n")
                source.insert(i+2, "    mask_up = F.interpolate(mask_t.float(), size=val_spectra_batch.shape[-1], mode='nearest')\n")
                source.insert(i+3, "    mask_np = mask_up.cpu().numpy()               # (B, 1, L)\n")
                break

with open('/home/y222/y222446/SpectraBreast/notebooks/02_pretraining.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
