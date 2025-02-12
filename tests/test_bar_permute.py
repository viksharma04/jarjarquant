import unittest
import pandas as pd
import numpy as np
from jarjarquant import BarPermute


class TestBarPermute(unittest.TestCase):

    def setUp(self):
        # Create sample OHLC data
        dates = pd.date_range('2023-01-01', periods=5)
        data1 = {
            'open': [1, 2, 3, 4, 5],
            'high': [2, 3, 4, 5, 6],
            'low': [0.5, 1.5, 2.5, 3.5, 4.5],
            'close': [1.5, 2.5, 3.5, 4.5, 5.5]
        }
        data2 = {
            'open': [10, 20, 30, 40, 50],
            'high': [20, 30, 40, 50, 60],
            'low': [5, 15, 25, 35, 45],
            'close': [15, 25, 35, 45, 55]
        }
        self.df1 = pd.DataFrame(data1, index=dates)
        self.df2 = pd.DataFrame(data2, index=dates)
        self.ohlc_df_list = [self.df1, self.df2]

    def test_initialization(self):
        bar_permute = BarPermute(self.ohlc_df_list)
        self.assertEqual(bar_permute.n_markets, 2)
        self.assertEqual(bar_permute.original_index.tolist(),
                         self.df1.index.tolist())
        self.assertEqual(len(bar_permute.rel_prices), 2)

    def test_permute(self):
        bar_permute = BarPermute(self.ohlc_df_list)
        shuffled_df_list = bar_permute.permute()
        self.assertEqual(len(shuffled_df_list), 2)
        for df in shuffled_df_list:
            self.assertEqual(len(df), len(self.df1))
            self.assertTrue(
                all(df.columns == ['open', 'high', 'low', 'close']))
            self.assertTrue(all(df.index == self.df1.index))

    def test_invalid_initialization(self):
        with self.assertRaises(ValueError):
            BarPermute([])

    def test_permute_length_mismatch(self):
        bar_permute = BarPermute(self.ohlc_df_list)
        # Modify the index length
        bar_permute.original_index = bar_permute.original_index[:-1]
        with self.assertRaises(ValueError):
            bar_permute.permute()


if __name__ == '__main__':
    unittest.main()
