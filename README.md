# XMR: An Explainable Multimodal Neural Network for Drug Response Prediction

Implementation of XMR as described in the paper "XMR: An Explainable Multimodal Neural Network for Drug Response Prediction"

<img src="https://github.com/zwa2/XMR/blob/main/images/XMR.png"  width=80% height=80%>

## Requirements
Networks have been implemented in Python 3.8 with the following packages:

* PyTorch
* networkx
* RDKit
* scikit-learn
* numpy

## Usage

To train an XMR model, follow these simple steps:

Clone our repository:

```
git clone https://github.com/zwa2/XMR.git
```

Load submodules:

```
git submodule update --init --recursive
```

Train the XMR model using the provided pathway:

```
sh train.sh
```

Improve the performance and reduce the model size by pruning the well-trained XMR model:

```
sh prune.sh
```

Additionally, you can modify the model's hyperparameters as described in `train.sh` and `prune.sh`.


