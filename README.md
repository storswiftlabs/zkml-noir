# zkML-Noir



## Description

Python ML model transcoding Noir, including various algorithms such as Decision tree, K-Means, XGBoost, FNN, CNN

## Directory structure

- data: ML model training and prediction data, including raw data and pre-processed data
- model: Three types of ML models
- noir: zkML generated Noir code
- tests: Testcase for zkML transpiler code
- zkml: The zkML code generation and floating point numbers quantize integer numbers
  - decision_tree: Python2Noir generate Noir prediction code for the decision tree based on sk-learn library
  - k_Means: Python2Noir generate Noir prediction code for the center points based on sk-learn library
  - quantization: ML floating point numbers quantize integer numbers
  - routine_code_generate: Routine generate Noir prediction code for the CNN and RNN based on Pytorch library
  - XGBoost: Python2Noir generate Noir prediction code for the XGBoost classification and regression based on XGBoost library
  
## Build guide

- Python 3.7+
- Anaconda

## Import package

- python2noir
- joblib
- scikit-learn
- xgboost
- numpy
- unittest
- pandas
- pytorch
- torchvision

## Usage
```shell
git clone https://github.com/storswiftlabs/zkml-noir.git
cd zkml-noir
# execute decision tree generate code
python  -m unittest tests/zkml/decision_tree/test_decision_tree_to_noir.py

# execute K-Means generate code
python  -m unittest tests/zkml/k_Means/test_k_Means_to_noir.py

# execute XGBoost generate code
python  -m unittest tests/zkml/XGBoost/test_xgboost_to_noir.py

# Train the CNN model
python tests/zkml/cnn/mnist_cnn.py
# execute FNN generate code
python zkml/routine_code_generate/fnn_to_noir.py
# execute CNN generate code
python zkml/routine_code_generate/cnn_to_noir.py
# Load the model and extract inputs
python zkml/routine_code_generate/extract_inputs.py
```
