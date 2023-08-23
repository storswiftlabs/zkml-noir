# zkML-Noir



## Description

Python implement ML model transcoding Noir contracts

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