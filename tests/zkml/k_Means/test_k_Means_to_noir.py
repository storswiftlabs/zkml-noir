import math
import unittest

import joblib
from transpiler.sub_module.primitive_type import FIELD

from k_Means.k_Means_to_noir import generate_k_means_noir_code, quantize_centers, generate_inputs


class test_k_Means_to_noir(unittest.TestCase):
    def test_main(self):
        model = joblib.load('../../../model/k_Means/kMeans.pkl')
        centers = model.cluster_centers_
        _scale_molecule, _scale_denominator, _zero_point, centers = quantize_centers(centers, 'uint8')
        print(math.ceil(_scale_molecule / _scale_denominator))
        noir_code = generate_k_means_noir_code(centers, math.ceil(_scale_molecule/_scale_denominator), _zero_point, FIELD)
        print(''.join(noir_code))


if __name__ == '__main__':
    unittest.main()
