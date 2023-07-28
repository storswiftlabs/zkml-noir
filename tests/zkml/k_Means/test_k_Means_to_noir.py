import unittest

import joblib

from k_Means.k_Means_to_noir import generate_k_means_noir_code


class test_k_Means_to_noir(unittest.TestCase):
    def test_main(self):
        model = joblib.load('kMeans.pkl')
        centers = model.cluster_centers_
        noir_code = generate_k_means_noir_code(centers, 'i64', 100)
        print(''.join(noir_code))


if __name__ == '__main__':
    unittest.main()
