import os
import unittest

import joblib
import pandas as pd

from XGBoost.xgboost_to_noir import xgboost_noir_code
from transpiler.context.noir_context import NoirContext
from utils.utils import quantize


class test_xgboost(unittest.TestCase):
    def test_xgboost_to_noir(self):
        # quantize
        path = "Acute_Inflammations"
        new_path = os.path.join(os.path.join('data', path), 'preprocessing_data.tsv')
        titanic = pd.read_table(new_path, sep="\t", header=None)
        fixed_number, is_negative = quantize(titanic.iloc[0])
        print("fixed_number is", fixed_number)
        # model to noir code
        model_path = "model/Acute_Inflammations_xgboost_classification.dat"
        is_classification = "classification" in model_path
        # load xgb model
        clf = joblib.load(model_path)
        noir_code = xgboost_noir_code(clf, 100, is_classification)
        with open(f"./{path}.nr", "w") as f:
            for line in noir_code:
                f.write(line)
