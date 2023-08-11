import math
import unittest
import joblib
import pandas as pd

from decision_tree.decision_tree_to_noir import generate_dt
from quantization.quantize import quantize_all, UINT, quantize_not_clip, calc_scale, calc_zero_point, quantize


class test_decision_tree_to_noir(unittest.TestCase):

    def test_all_preprocess_data_main(self):
        _type = "uint8"
        path = 'data/Acute_Inflammations/preprocessing_data.tsv'
        titanic = pd.read_table(path, sep="\t", header=None)
        titanic = titanic.iloc[:, :-1]
        print(titanic.values.tolist())
        x = [element for sublist in titanic.values.tolist() for element in sublist]
        # print(titanic)
        scale_molecule, scale_denominator = calc_scale(x, _type)
        scale = math.ceil(scale_molecule / scale_denominator)
        print(scale)
        _zero_point = calc_zero_point(x, scale, _type)
        print(_zero_point)
        x = quantize(x, scale, _zero_point, _type)
        new_data = [x[i:i + titanic.shape[1]] for i in range(0, len(x), titanic.shape[1])]
        print(new_data)
        # print("quantize", quantize(x, scale, _zero_point, _type))

    def test_main(self):
        model = joblib.load('model/decision_tree/dt.pkl')
        # quantize part
        _type = "uint8"
        path = 'data/Acute_Inflammations/preprocessing_data.tsv'
        titanic = pd.read_table(path, sep="\t", header=None)
        titanic = titanic.iloc[:, :-1]
        print(titanic.values.tolist())
        x = [element for sublist in titanic.values.tolist() for element in sublist]
        # print(titanic)
        scale_molecule, scale_denominator = calc_scale(x, _type)
        q_scale = math.ceil(scale_molecule / scale_denominator)
        q_zero_point = calc_zero_point(x, q_scale, _type)
        print("quantization info: ", q_scale, q_zero_point)

        # deal with user inputs
        noir_inputs_data = [0.9333333333333336, 1.0, 1.0, 1.0, 1.0, 0.0]
        print("noir_inputs_data", quantize_not_clip(noir_inputs_data, q_scale, q_zero_point))

        # generate noir code
        noir_code = generate_dt(model, True, q_scale, q_zero_point, UINT[0])
        print(''.join(noir_code))


if __name__ == '__main__':
    unittest.main()
