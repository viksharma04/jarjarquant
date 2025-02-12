import unittest
import pandas as pd
import numpy as np
from jarjarquant import BarPermute, PricePermute


class TestPricePermute(unittest.TestCase):

    def setUp(self):
        # Create sample price series data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data1 = np.cumsum(np.random.randn(100))
        data2 = np.cumsum(np.random.randn(100))
        self.series1 = pd.Series(data1, index=dates)
        self.series2 = pd.Series(data2, index=dates)
        self.price_series_list = [self.series1, self.series2]

    def test_initialization(self):
        # Test initialization of PricePermute
        pp = PricePermute(self.price_series_list)
        self.assertEqual(pp.n_markets, 2)
        self.assertEqual(len(pp.price_series_list), 2)
        self.assertEqual(pp.price_series_list[0].iloc[0], self.series1.iloc[0])
        self.assertEqual(pp.price_series_list[1].iloc[0], self.series2.iloc[0])

    def test_initialization_with_empty_list(self):
        # Test initialization with empty list
        with self.assertRaises(ValueError):
            PricePermute([])

    def test_initialization_with_different_lengths(self):
        # Test initialization with series of different lengths
        series3 = self.series1[:-1]
        with self.assertRaises(ValueError):
            PricePermute([self.series1, series3])

    def test_permute(self):
        # Test permute method
        pp = PricePermute(self.price_series_list)
        shuffled_series_list = pp.permute()
        self.assertEqual(len(shuffled_series_list), 2)
        self.assertEqual(len(shuffled_series_list[0]), 100)
        self.assertEqual(len(shuffled_series_list[1]), 100)
        self.assertTrue(
            (shuffled_series_list[0].index == self.series1.index).all())
        self.assertTrue(
            (shuffled_series_list[1].index == self.series2.index).all())


class TestBarPermute(unittest.TestCase):
    """
    Unit tests for the BarPermute class.
    TestBarPermute is a unittest.TestCase subclass that tests the functionality of the BarPermute class.
    It includes the following test methods:
    - setUp: Initializes sample OHLC data for testing.
    - test_initialization: Tests the initialization of the BarPermute class.
    - test_permute: Tests the permute method of the BarPermute class.
    - test_invalid_initialization: Tests the initialization of BarPermute with an empty list.
    - test_permute_length_mismatch: Tests the permute method when there is a length mismatch in the index.
    """

    def setUp(self):
        # Create sample OHLC data
        dates = pd.date_range('2023-01-01', periods=5)
        data1 = {
            'Open': [1, 2, 3, 4, 5],
            'High': [2, 3, 4, 5, 6],
            'Low': [0.5, 1.5, 2.5, 3.5, 4.5],
            'Close': [1.5, 2.5, 3.5, 4.5, 5.5]
        }
        data2 = {
            'Open': [10, 20, 30, 40, 50],
            'High': [20, 30, 40, 50, 60],
            'Low': [5, 15, 25, 35, 45],
            'Close': [15, 25, 35, 45, 55]
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
                all(df.columns == ['Open', 'High', 'Low', 'Close']))
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
