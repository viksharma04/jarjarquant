"Initialize jarjarquant"
try:
    from .feature_engineering import FeatureEngineering
    from .labelling import Labelling
except ImportError as e:
    from feature_engineering import FeatureEngineering
    from labelling import Labelling

__version__ = "0.1.0"
