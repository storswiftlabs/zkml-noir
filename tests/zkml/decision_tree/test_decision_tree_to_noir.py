import unittest
import joblib

from decision_tree.decision_tree_to_noir import generate_dt


class test_decision_tree_to_noir(unittest.TestCase):

    def test_main(self):
        # mode_path = 'model/dt.pkl'
        model = joblib.load('dt.pkl')
        noir_code = generate_dt(model, 10, True)

        print(''.join(noir_code))


if __name__ == '__main__':
    unittest.main()
