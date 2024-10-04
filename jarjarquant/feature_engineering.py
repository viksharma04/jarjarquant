import pandas as pd
import numpy as np

class FeatureEngineering:
    """Class to implement common featrue engineering transformations"""

    def __init__(self, features_df):
        """Initialize Featureengineering

        Args:
            features_df (pd.DataFrame): a pandas dataframe containg the features timeseries with pd.DateTime index
        """
        self.features_df = features_df

    def getWeights(d, size):

        w=[1.]
        for k in range(1,size):
            w_ = -w[-1]/k*(d-k+1)
            w.append(w_)
        w=np.array(w[::-1]).reshape(-1,1)
        return w
