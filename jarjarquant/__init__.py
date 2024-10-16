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
    """_summary_

    Args:
        DataGatherer (_type_): _description_
        Labeller (_type_): _description_
        FeatureEngineer (_type_): _description_
    """

    def __init__(self, series=None):

        super().__init__()
        self._series = series
        self.labeller = Labeller(timeseries=series)

    @classmethod
    def from_random_normal(cls, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, **kwags):
        """Create a random price series using returns from a normal distribution

        Args:
            loc (float, optional): mean return. Defaults to 0.005.
            volatility (float, optional): period volatility. Defaults to 0.05.
            periods (int, optional): number of data points. Defaults to 252.

        Returns:
            _type_: _description_
        """

        cls.data_gatherer = DataGatherer()
        series = cls.data_gatherer.generate_random_normal(
            loc=loc, volatility=volatility, periods=periods, **kwags)

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

    # TODO: Repl function
