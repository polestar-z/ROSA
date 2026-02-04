# ROSA

RoSA: Role-Separated Attention for Multi-label Heterogeneous Graphs

## Environment
- Python 3.8+
- Dependencies: requirements.txt

## Run
1. From the repo root
2. Prepare the dataset (see Dataset)
3. Run training

```bash
python main.py -m ROSA -d dataset -t node_classification -g 1 --multi_label
```

- dataset optional: imdb_node_classification / amazon_node_classification / cite_node_classification
- Single GPU: set `gpu=0` (or another index)
- Logs and TensorBoard output go to `/output/<model_name>/`
- models optional: ROSA / AOA_Multi / COA_Multi

## Config
- Default hyperparameters live in `config.ini`
- You can override them directly in `Experiment(...)` (for example `max_epoch=10`)

## Dataset

- https://huggingface.co/datasets/kg4sci/ROSA/tree/main

- place it on /ROSA/dataset