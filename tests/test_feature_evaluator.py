from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from jarjarquant.data_service import DataService, SampleRequest
from jarjarquant.feature_evaluator import FeatureEvaluator
from jarjarquant.indicators.base import IndicatorSpec
from jarjarquant.indicators.registry import IndicatorType


def test_indicator_threshold_search_linear():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function with linear threshold option
    result = FeatureEvaluator.indicator_threshold_search(
        indicator_values=indicator_values,
        associated_returns=associated_returns,
        n_thresholds=10,
        threshold_option="linear",
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = [
        "Threshold",
        "% values > threshold",
        "Spearman correlation above threshold",
        "Mean return above threshold",
        "Std dev return above threshold",
        "Median return above threshold",
        "Q25 return above threshold",
        "Q75 return above threshold",
        "PF Long above threshold",
        "PF Short above threshold",
        "% values < threshold",
        "Spearman correlation below threshold",
        "Mean return below threshold",
        "Std dev return below threshold",
        "Median return below threshold",
        "Q25 return below threshold",
        "Q75 return below threshold",
        "PF Long below threshold",
        "PF Short below threshold",
    ]
    assert list(result.columns) == expected_columns


def test_indicator_threshold_search_percentile():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function with percentile threshold option
    result = FeatureEvaluator.indicator_threshold_search(
        indicator_values=indicator_values,
        associated_returns=associated_returns,
        n_thresholds=10,
        threshold_option="percentile",
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = [
        "Threshold",
        "% values > threshold",
        "Spearman correlation above threshold",
        "Mean return above threshold",
        "Std dev return above threshold",
        "Median return above threshold",
        "Q25 return above threshold",
        "Q75 return above threshold",
        "PF Long above threshold",
        "PF Short above threshold",
        "% values < threshold",
        "Spearman correlation below threshold",
        "Mean return below threshold",
        "Std dev return below threshold",
        "Median return below threshold",
        "Q25 return below threshold",
        "Q75 return below threshold",
        "PF Long below threshold",
        "PF Short below threshold",
    ]
    assert list(result.columns) == expected_columns


def test_indicator_threshold_search_predefined_thresholds():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))
    predefined_thresholds = [20, 40, 60, 80]

    # Call the function with predefined thresholds
    result = FeatureEvaluator.indicator_threshold_search(
        indicator_values=indicator_values,
        associated_returns=associated_returns,
        thresholds=predefined_thresholds,
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = [
        "Threshold",
        "% values > threshold",
        "Spearman correlation above threshold",
        "Mean return above threshold",
        "Std dev return above threshold",
        "Median return above threshold",
        "Q25 return above threshold",
        "Q75 return above threshold",
        "PF Long above threshold",
        "PF Short above threshold",
        "% values < threshold",
        "Spearman correlation below threshold",
        "Mean return below threshold",
        "Std dev return below threshold",
        "Median return below threshold",
        "Q25 return below threshold",
        "Q75 return below threshold",
        "PF Long below threshold",
        "PF Short below threshold",
    ]
    assert list(result.columns) == expected_columns


def test_indicator_threshold_search_invalid_threshold_option():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function with an invalid threshold option
    with pytest.raises(
        ValueError, match="threshold_option must be 'linear' or 'percentile'."
    ):
        FeatureEvaluator.indicator_threshold_search(
            indicator_values=indicator_values,
            associated_returns=associated_returns,
            n_thresholds=10,
            threshold_option="invalid_option",
        )


def test_indicator_threshold_search_no_thresholds_provided():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function without providing thresholds or n_thresholds
    with pytest.raises(
        ValueError, match="Either n_thresholds or thresholds must be provided."
    ):
        FeatureEvaluator.indicator_threshold_search(
            indicator_values=indicator_values,
            associated_returns=associated_returns,
            thresholds=None,
            n_thresholds=None,
        )


class TestDataServiceDatabase:
    def test_save_to_database_with_polars_dataframe(self):
        """Test saving a Polars DataFrame to database."""
        # Create test data
        data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Mock the DataService
        ds = DataService()
        mock_db_file_path = MagicMock()
        mock_db_file_path.parent.mkdir = MagicMock()
        ds.db_path = MagicMock()
        ds.db_path.__truediv__ = MagicMock(return_value=mock_db_file_path)

        mock_conn = MagicMock()
        with patch("duckdb.connect", return_value=mock_conn) as mock_connect:
            with patch.object(ds, "_table_exists", return_value=False):
                ds.save_to_database(data, "test_table", append=False)
                mock_connect.assert_called_once_with(mock_db_file_path)
                mock_conn.execute.assert_called_once()

    def test_save_to_database_with_pandas_dataframe(self):
        """Test saving a Pandas DataFrame to database."""
        # Create test data
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Mock the DataService
        ds = DataService()
        mock_db_file_path = MagicMock()
        mock_db_file_path.parent.mkdir = MagicMock()
        ds.db_path = MagicMock()
        ds.db_path.__truediv__ = MagicMock(return_value=mock_db_file_path)

        mock_conn = MagicMock()
        with patch("duckdb.connect", return_value=mock_conn) as mock_connect:
            with patch.object(ds, "_table_exists", return_value=False):
                ds.save_to_database(data, "test_table", append=False)
                mock_connect.assert_called_once_with(mock_db_file_path)
                mock_conn.execute.assert_called_once()

    def test_save_to_database_empty_data(self):
        """Test that saving empty data raises ValueError."""
        ds = DataService()

        with pytest.raises(ValueError, match="Cannot save empty or None data"):
            ds.save_to_database(pl.DataFrame(), "test_table")

        # Note: We use type: ignore to suppress type checker warnings
        # since the method actually handles None data at runtime
        with pytest.raises(ValueError, match="Cannot save empty or None data"):
            ds.save_to_database(None, "test_table")  # type: ignore

    def test_load_from_database_existing_file(self):
        """Test loading data from existing database file."""
        from pathlib import Path

        ds = DataService()
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        ds.db_path = MagicMock()
        ds.db_path.__truediv__.return_value = mock_path

        expected_data = pl.DataFrame({"col1": [1, 2, 3]})

        mock_conn = MagicMock()
        mock_conn.execute.return_value.pl.return_value = expected_data

        with patch("duckdb.connect", return_value=mock_conn):
            with patch.object(ds, "_table_exists", return_value=True):
                result = ds.load_from_database("test_table")
                assert result is not None
                assert result.equals(expected_data)

    def test_load_from_database_nonexistent_file(self):
        """Test loading data from non-existent database file returns None."""
        from pathlib import Path

        ds = DataService()
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        ds.db_path = MagicMock()
        ds.db_path.__truediv__.return_value = mock_path

        result = ds.load_from_database("test_table")
        assert result is None


class TestParallelIndicatorDistributionStudy:
    @patch(
        "jarjarquant.feature_evaluator.FeatureEvaluator.indicator_distribution_study"
    )
    def test_parallel_indicator_distribution_study_save_run_false(self, mock_study):
        """Test parallel_indicator_distribution_study with save_run=False."""
        # Mock the static method to return basic outputs
        mock_study.return_value = {"basic_outputs": [True, False, 0.5, 1.2]}

        # Create test data
        fe = FeatureEvaluator()
        fe.ds = MagicMock()

        # Mock sample data
        sample_data = pl.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "date": ["2023-01-01", "2023-01-01"],
                "Open": [100, 200],
                "High": [105, 205],
                "Low": [95, 195],
                "Close": [102, 202],
                "Volume": [1000, 2000],
            }
        )

        mock_sample = MagicMock()
        mock_sample.data = sample_data
        fe.ds.get_sample.return_value = mock_sample

        # Create mock indicator spec
        indicator_spec = IndicatorSpec(IndicatorType.RSI, {"period": 14})
        sample_request = SampleRequest(
            "equities", "2023-01-01", "2023-01-02", n_samples=2
        )

        with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.map.return_value = [
                {"basic_outputs": [True, False, 0.5, 1.2]},
                {"basic_outputs": [False, True, 0.7, 0.8]},
            ]

            result = fe.parallel_indicator_distribution_study(
                indicator_spec, sample_request, save_run=False
            )

            # Verify results
            assert "ADF Test" in result
            assert "Jarque-Bera Test" in result
            assert "Relative Entropy" in result
            assert "Range-IQR Ratio" in result

    @patch("jarjarquant.feature_evaluator.FeatureEvaluator._save_detailed_results")
    @patch(
        "jarjarquant.feature_evaluator.FeatureEvaluator.indicator_distribution_study"
    )
    def test_parallel_indicator_distribution_study_save_run_true(
        self, mock_study, mock_save
    ):
        """Test parallel_indicator_distribution_study with save_run=True."""
        # Mock detailed outputs
        mock_eval_result = MagicMock()
        mock_study.return_value = {
            "basic_outputs": [True, False, 0.5, 1.2],
            "ticker": "AAPL",
            "eval_result": mock_eval_result,
        }

        # Create test data
        fe = FeatureEvaluator()
        fe.ds = MagicMock()

        # Mock sample data
        sample_data = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": ["2023-01-01"],
                "Open": [100],
                "High": [105],
                "Low": [95],
                "Close": [102],
                "Volume": [1000],
            }
        )

        mock_sample = MagicMock()
        mock_sample.data = sample_data
        fe.ds.get_sample.return_value = mock_sample

        # Create mock indicator spec
        indicator_spec = IndicatorSpec(IndicatorType.RSI, {"period": 14})
        sample_request = SampleRequest(
            "equities", "2023-01-01", "2023-01-02", n_samples=1
        )

        with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.map.return_value = [
                {
                    "basic_outputs": [True, False, 0.5, 1.2],
                    "ticker": "AAPL",
                    "eval_result": mock_eval_result,
                }
            ]

            result = fe.parallel_indicator_distribution_study(
                indicator_spec, sample_request, save_run=True
            )

            # Verify that save was called
            mock_save.assert_called_once()

            # Verify results
            assert "ADF Test" in result
            assert "Jarque-Bera Test" in result

    def test_save_detailed_results(self):
        """Test _save_detailed_results method."""
        fe = FeatureEvaluator()
        fe.ds = MagicMock()

        # Create mock data
        mock_adf = MagicMock()
        mock_adf.statistic = 1.5
        mock_adf.pvalue = 0.05
        mock_adf.lags = 2
        mock_adf.nobs = 100
        mock_adf.critical_values = {"1%": -3.5, "5%": -2.9}
        mock_adf.decision = "Stationary"
        mock_adf.is_stationary = True

        mock_jb = MagicMock()
        mock_jb.statistic = 2.0
        mock_jb.pvalue = 0.1
        mock_jb.method = "Jarque-Bera"
        mock_jb.decision = "Normal"
        mock_jb.is_normal = True

        mock_entropy = MagicMock()
        mock_entropy.entropy = 0.8
        mock_entropy.normalized_entropy = 0.7
        mock_entropy.n_bins = 10
        mock_entropy.n_observations = 100
        mock_entropy.quality_assessment = "Good"
        mock_entropy.is_concerning = False

        mock_range_iqr = MagicMock()
        mock_range_iqr.ratio = 1.5
        mock_range_iqr.range_value = 10.0
        mock_range_iqr.iqr_value = 6.7
        mock_range_iqr.q25 = 2.5
        mock_range_iqr.q75 = 9.2
        mock_range_iqr.min_value = 0.0
        mock_range_iqr.max_value = 10.0
        mock_range_iqr.n_observations = 100

        mock_eval_result = MagicMock()
        mock_eval_result.adf_test = mock_adf
        mock_eval_result.jb_normality_test = mock_jb
        mock_eval_result.relative_entropy = mock_entropy
        mock_eval_result.range_iqr_ratio = mock_range_iqr

        results = [
            {
                "ticker": "AAPL",
                "eval_result": mock_eval_result,
                "basic_outputs": [True, True, 0.8, 1.5],
            }
        ]

        # Mock indicator spec
        indicator_spec = MagicMock()
        indicator_spec.indicator_type.value = "RSI"
        indicator_spec.parameters = {"period": 14}

        # Mock sample request
        sample_request = MagicMock()
        sample_request.bar_size.value = "1d"
        sample_request.start_date = "2023-01-01"
        sample_request.end_date = "2023-01-02"

        # Test the method
        fe._save_detailed_results(results, sample_request, indicator_spec)

        # Verify save_to_database was called with correct arguments
        fe.ds.save_to_database.assert_called_once()
        args, kwargs = fe.ds.save_to_database.call_args
        saved_df = args[0]
        table_name = args[1]

        assert table_name == "ind_dist_studies_db"
        assert kwargs.get("append") is True
        assert isinstance(saved_df, pl.DataFrame)
        assert saved_df.height == 1  # One row

        # Check that required columns are present
        expected_columns = [
            "indicator",
            "ticker",
            "bar_size",
            "start_date",
            "end_date",
            "indicator_params",
            "timestamp",
            "adf_statistic",
            "adf_pvalue",
        ]
        for col in expected_columns:
            assert col in saved_df.columns

    def test_save_detailed_results_empty_results(self):
        """Test _save_detailed_results with empty results."""
        fe = FeatureEvaluator()
        fe.ds = MagicMock()

        indicator_spec = MagicMock()
        sample_request = MagicMock()

        # Test with empty results
        fe._save_detailed_results([], sample_request, indicator_spec)

        # Verify save_to_database was not called
        fe.ds.save_to_database.assert_not_called()

    def test_save_detailed_results_invalid_results(self):
        """Test _save_detailed_results with invalid results structure."""
        fe = FeatureEvaluator()
        fe.ds = MagicMock()

        indicator_spec = MagicMock()
        sample_request = MagicMock()

        # Test with invalid results (missing eval_result)
        invalid_results = [{"ticker": "AAPL", "basic_outputs": [1, 2, 3, 4]}]

        fe._save_detailed_results(invalid_results, sample_request, indicator_spec)

        # Verify save_to_database was not called
        fe.ds.save_to_database.assert_not_called()
