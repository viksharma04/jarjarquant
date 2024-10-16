import unittest
import pandas as pd
import numpy as np
from labeller import Labeller


class TestLabeller(unittest.TestCase):

    def setUp(self):
        """Initialize before each test."""
        self.series = pd.Series([100, 101, 103, 97, 99, 101], index=pd.date_range(
            start="2023-01-01", periods=6))
        self.labeller = Labeller(self.series)

    def test_initialization(self):
        """Test initialization and datetime index validation."""
        self.assertIsInstance(self.labeller.series, pd.Series)
        with self.assertRaises(ValueError):
            # No datetime index
            Labeller(pd.Series([1, 2, 3], index=[1, 2, 3]))

    def test_inverse_cumsum_filter(self):
        """Test the inverse_cumsum_filter method."""
        h = 0.01
        n = 2
        flagged = self.labeller.inverse_cumsum_filter(self.series, h=h, n=n)
        self.assertIsInstance(flagged, pd.Series)
        # Test expected number of flagged entries
        self.assertEqual(flagged.sum(), 4)

    def test_plot_with_flags(self):
        """Test the plot_with_flags method."""
        flagged = self.labeller.inverse_cumsum_filter(self.series, h=0.01, n=2)
        # Ensure plotting doesn't raise exceptions
        try:
            self.labeller.plot_with_flags(self.series, flagged)
        except Exception as e:
            self.fail(f"plot_with_flags raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
