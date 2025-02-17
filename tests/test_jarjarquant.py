# import unittest
# import pandas as pd
# from jarjarquant import Jarjarquant


# class TestJarjarquant(unittest.TestCase):

#     def setUp(self):
#         """Initialize before each test."""
#         self.default_series = pd.Series(
#             [100, 101, 102, 103, 105], index=pd.date_range(start="2023-01-01", periods=5))
#         self.jq = Jarjarquant(self.default_series)

#     def test_initialization(self):
#         """Test the Jarjarquant class initialization."""
#         self.assertIsInstance(self.jq.series, pd.Series)
#         self.assertEqual(len(self.jq.series), 5)

#     def test_from_random_normal(self):
#         """Test the from_random_normal method."""
#         jq_random = Jarjarquant.from_random_normal(
#             loc=0.005, volatility=0.05, periods=100)
#         self.assertIsInstance(jq_random.series, pd.Series)
#         self.assertEqual(len(jq_random.series), 100)

#     def test_from_yf_ticker(self):
#         """Test the from_yf_ticker method."""
#         jq_ticker = Jarjarquant.from_yf_ticker(ticker="SPY")
#         self.assertIsInstance(jq_ticker.series, pd.Series)

#     def test_series_setter(self):
#         """Test the series setter and validation."""
#         new_series = pd.Series([200, 201, 202], index=pd.date_range(
#             start="2023-02-01", periods=3))
#         self.jq.series = new_series
#         self.assertEqual(self.jq.series.iloc[0], 200)

#         # Test invalid input
#         with self.assertRaises(ValueError):
#             self.jq.series = [1, 2, 3]

#     def test_series_deleter(self):
#         """Test deleting the series."""
#         del self.jq.series
#         self.assertFalse(hasattr(self.jq, '_series'))


# if __name__ == '__main__':
#     unittest.main()
