import math
import os
import unittest

import joblib
import pandas as pd

from XGBoost.xgboost_to_noir import xgboost_noir_code
from transpiler.context.noir_context import NoirContext

from quantization.quantize import calc_scale, calc_zero_point
from utils.utils import quantize


class test_xgboost(unittest.TestCase):
    def test_xgboost_to_noir(self):
        # quantize part
        q_type = "uint8"
        path = 'data/Acute_Inflammations/preprocessing_data.tsv'
        titanic = pd.read_table(path, sep="\t", header=None)
        titanic = titanic.iloc[:, :-1]
        print("normalization processing data info: ", titanic.values.tolist())
        x = [element for sublist in titanic.values.tolist() for element in sublist]

        scale_molecule, scale_denominator = calc_scale(x, q_type)
        q_scale = math.ceil(scale_molecule / scale_denominator)
        q_zero_point = calc_zero_point(x, q_scale, q_type)
        print("quantization info: ", q_scale, q_zero_point)

        # quantize
        path = "Acute_Inflammations"
        new_path = os.path.join(os.path.join('data', path), 'preprocessing_data.tsv')
        titanic = pd.read_table(new_path, sep="\t", header=None)

        # model to noir code
        # model_path = "model/XGBoost/Acute_Inflammations_xgboost_classification.dat"
        model_path = "model/XGBoost/Acute_Inflammations_xgboost_regression.dat"
        is_classification = "classification" in model_path

        # load xgb model
        clf = joblib.load(model_path)
        noir_code = xgboost_noir_code(clf, q_scale, q_zero_point, q_type, is_classification)
        with open(f"./{path}.nr", "w") as f:
            for line in noir_code:
                f.write(line)
