# MLOps Lab 2 — Automating Dataset Extension (CIFAR-10)
Configuration-driven system that dynamically builds training and validation sets from CIFAR-10 data batches while keeping the test set completely static.

### Experiments
- 1 batch   -> 10,000 train images
- 2 batches -> 20,000 train images  
- 3 batches -> 30,000 train images
- Full      -> 40,000 train images

### Installation

```bash
poetry install
```
### How to Run
```bash
poetry run python -m src.train --config configs/config_1batch.yaml
poetry run python -m src.train --config configs/config_2batches.yaml
poetry run python -m src.train --config configs/config_3batches.yaml
poetry run python -m src.train --config configs/config_full.yaml
```
Results (metrics + Train/Val Loss plots) are saved in the `results/` folder.