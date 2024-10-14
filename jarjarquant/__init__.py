"Initialize jarjarquant"
try:
    from .feature_engineering import FeatureEngineering
except ImportError as e:
    from feature_engineering import FeatureEngineering

__version__ = "0.0.1"
