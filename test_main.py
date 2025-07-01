import unittest
import pandas as pd
import numpy as np
from main import FunctionSelector, TestMapper, CustomException

class TestFunctionLogic(unittest.TestCase):

    def setUp(self):
        # Create mock data
        x_vals = np.linspace(-1, 1, 5)
        self.train_df = pd.DataFrame({
            'x': x_vals,
            'y1': np.sin(x_vals),
            'y2': np.cos(x_vals),
            'y3': x_vals ** 2,
            'y4': x_vals * 2
        })
        self.ideal_df = pd.DataFrame({'x': x_vals})
        for i in range(1, 51):
            self.ideal_df[f'y{i}'] = np.sin(x_vals) + (i - 1) * 0.01
        self.test_df = pd.DataFrame({'x': x_vals, 'y': np.sin(x_vals)})

        self.train_df.to_csv("mock_train.csv", index=False)
        self.ideal_df.to_csv("mock_ideal.csv", index=False)
        self.test_df.to_csv("mock_test.csv", index=False)

    def test_select_best_functions(self):
        selector = FunctionSelector("mock_train.csv", "mock_ideal.csv", "mock_test.csv")
        best_matches = selector.select_best_functions()
        self.assertEqual(len(best_matches), 4)
        for func in best_matches.values():
            self.assertIn(func, self.ideal_df.columns)

    def test_map_test_points(self):
        selector = FunctionSelector("mock_train.csv", "mock_ideal.csv", "mock_test.csv")
        best_matches = selector.select_best_functions()
        mapper = TestMapper("mock_train.csv", "mock_ideal.csv", "mock_test.csv", best_matches)
        results = mapper.map_test_points()
        self.assertTrue(all(col in results.columns for col in ['x', 'y', 'delta_y', 'ideal_func']))

if __name__ == '__main__':
    unittest.main()
