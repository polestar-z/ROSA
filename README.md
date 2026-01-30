# ROSA

基于 DGL 的异构图节点分类实验代码。

## 环境准备
- Python 3.8+（建议）
- 依赖：`torch`、`dgl`、`numpy`、`pandas`、`scikit-learn`、`tqdm`、`tensorboard`、`colorlog`、`colorama`
- 安装示例（请根据 CUDA 版本选择合适的 torch/dgl 版本）：

```bash
pip install torch dgl numpy pandas scikit-learn tqdm tensorboard colorlog colorama
```

## 运行
1. 进入仓库根目录
2. 准备好数据集（见 Dataset 部分）
3. 运行训练

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

- 单机单卡：将 `gpu` 改为 `0`/`1` 等即可
- 日志与 TensorBoard 输出默认在 `./openhgnn/output/<model_name>/`

## 配置
- 默认超参在 `config.ini`
- 也可以在 `Experiment(...)` 里直接覆盖（例如 `max_epoch=10`）

## Dataset
