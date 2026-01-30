# ROSA

Heterogeneous graph node classification experiments based on DGL.

## Environment
- Python 3.8+ (recommended)
- Dependencies: torch, dgl, numpy, pandas, scikit-learn, tqdm, tensorboard, colorlog, colorama
- Install example (pick torch/dgl builds that match your CUDA):

```bash
pip install torch dgl numpy pandas scikit-learn tqdm tensorboard colorlog colorama
```

## Run
1. From the repo root
2. Prepare the dataset (see Dataset)
3. Run training

```bash
export PYTHONPATH="$(pwd)/..:$PYTHONPATH"
python - <<'PY'
from ROSA import Experiment

Experiment(
    model="ROSA",
    dataset="imdb_node_classification",
    task="node_classification",
    gpu=-1,
    use_distributed=False,
).run()
PY
```

- Single GPU: set `gpu=0` (or another index)
- Logs and TensorBoard output go to `openhgnn/output/<model_name>/`

## Config
- Default hyperparameters live in `config.ini`
- You can override them directly in `Experiment(...)` (for example `max_epoch=10`)

## Dataset
