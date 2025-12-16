# ML_charged_defects

This repository stores the data and programs that are used in the following paper:


[Shin Kiyohara, Chisa Shibui, Soungmin Bae, and Yu Kumagai
Phys. Rev. Lett. 135, 246101, (2025).](https://doi.org/10.1103/h66h-y5k6)


# How to use
git clone https://github.com/kumagai-group/ML_charged_defects

There are four types of data in `oxy_vac_data` directory.
- materials_coreAlign: Oxygen vacancy formation energies, of which Fermi levels are aligned at the oxygen core potentials. Note that the results with perturbed host states (PHS) are excluded.
- materials_coreAlign_with_PHS: Same with materials_coreAlign but include results with PHS.
- materials_vbmAlign: Same with materials_coreAlign but the Fermi levels are aligned at the valence band maxima (VBM).
- materials_VBM: VBM positions with respect to the oxygen corepotentials.

These json file can be converted to Material class instance by

```
from monty.serialization import loadfn

loadfn( Path(oxy_vac_data)/ "defect_structure_info.json")
```

We can construct MaterialsDataset in materials_dataset.py from these data.

