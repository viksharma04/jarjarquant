"Initialize jarjarquant"
try:
    from .data_gatherer import DataGatherer
    from .feature_engineer import FeatureEngineer
    from .labeller import Labeller
except ImportError as e:
    from jarjarquant.data_gatherer import DataGatherer
    from jarjarquant.feature_engineer import FeatureEngineer
    from jarjarquant.labeller import Labeller

__version__ = "0.1.0"


class Jarjarquant(DataGatherer, Labeller, FeatureEngineer):

    def __init__(self, series=None):

        super().__init__()
        self._series = series
        self.labeller = Labeller(timeseries=series)

    @classmethod
    def from_random_normal(cls, **kwags):

        cls.data_gatherer = DataGatherer()
        series = cls.data_gatherer.generate_random_normal(**kwags)

        return cls(series)

    @classmethod
    def from_yf_ticker(cls, ticker: str = "SPY", **kwags):

        cls.data_gatherer = DataGatherer()
        series = cls.data_gatherer.get_yf_ticker(ticker, **kwags)

        return cls(series)

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, series):
        self._series = series

    @series.deleter
    def series(self):
        del self._series
