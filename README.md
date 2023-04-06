# Missing Data Imputation with Graph Laplacian Pyramid Network

In this paper, we propose a Graph Laplacian Pyramid Network (GLPN) for general imputation tasks, which follows the "draft-then-refine" procedures. Our model shows superior performance over state-of-art methods on three imputation tasks. 

## Installation

### Install via Conda and Pip

```
conda create --name GLPN python=3.8
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric
pip install -r requirements.txt
```

## Datasets

The datasets manipulated in this code can be downloaded to the following locations:

- the METR-LA traffic data: https://github.com/liyaguang/DCRNN;
- the NREL solar energy: https://www.nrel.gov/grid/solar-power-data.html
- the SeData traffic data: https://github.com/zhiyongc/Seattle-Loop-Data
- the PeMS traffic data: https://github.com/liyaguang/DCRNN
- the Texas, Cornell, ArXiv-Year, and YelpChi datasets: https://github.com/CUAI/Non-Homophily-Large-Scale
- the SYNTHIE, PROTEINS, and FRANKENSTEIN datasets: https://chrsmrrs.github.io/datasets/docs/datasets/

## Files

* `run_sensor_MCAR_MAR.py`  : train models under missing mechanisms of Missing Completely At Random (MCAR) and Missing Completely At Random (MAR) on continuous sensor datasets.
* `run_sensor_MNAR.py` : train models under missing mechanisms of Missing Not At Random (MNAR) on continuous sensor datasets.
* `run_single_graph.py` : train models under missing mechanisms of MCAR on single-graph datasets.
* `run_multi_graph.py` : train models under missing mechanisms of MCAR on multi-graph datasets
* `utils.py, dataset.py,data_utils.py ` :  data preprocessing; generate masks 
* `model_structure.py` : implementation of models
* `layer.py` : implementation of basic layers
* `eval_mert_MCAR.py` : an evaluation example on METR-LA dataset under MCAR setting
* `KNN_MCAR.py`,  `KNN_MAR.py`, `KNN_MNAR.py` : compute the metrics for the `MEAN` and `KNN` imputation methods

## Training

To train our model in the paper, run these commands:

```train
MCAR:
python run_sensor_MCAR_MAR.py --dataset metr --miss_rate 0.2 --setting MCAR
python run_single_graph.py --dataset cornell --miss_rate 0.2 --setting MCAR
python run_multi_graph.py --dataset cornell --miss_rate 0.2 --setting MCAR
MAR:
python run_sensor_MCAR_MAR.py --dataset FRANKENSTEIN --miss_rate 0.2 --setting MAR
MNAR:
python run_sensor_MNAR.py --dataset metr --miss_rate 0.2 
```

To train GCN baseline, run these commands:

```GCN baseline
python run_sensor_MCAR_MAR.py --dataset metr --miss_rate 0.2 --setting MCAR --model_name GCN_b
python run_sensor_MNAR.py --dataset metr --miss_rate 0.2 --setting MNAR --model_name GCN_b
```

To test Mean and kNN baselines, run these commands:

```mean,knn baseline
python KNN_MCAR.py
python KNN_MAR.py
python KNN_MNAR.py
```

## Evaluation

To evaluate my model on the METR-LA dataset, run:

```Evaluation
python eval_mert_MCAR.py  --dataset metr --miss_rate 0.2 --setting MCAR 
```

