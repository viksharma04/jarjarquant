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

    @staticmethod
    def get_weights(d, size):

        w=[1.]
        for k in range(1,size):
            w_ = -w[-1]/k*(d-k+1)
            w.append(w_)
        w=np.array(w[::-1]).reshape(-1,1)
        return w
    
    def frac_diff(self, d, thres=.01):

        w=self.get_weights(d, self.features_df.shape[0])
        w_=np.cumsum(abs(w))
        w_/=w_[-1]
        skip = w_[w_>thres].shape[0]
        df={}
        for name in self.features_df.columns:
            seriesF,df_=self.features_df[[name]].fillna(method='ffill').dropna(),pd.Series(index=self.features_df.index, dtype=float)
            for iloc in range(skip,seriesF.shape[0]):
                loc=seriesF.index[iloc]
                if not np.isfinite(self.features_df.loc[loc,name]): continue
                df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
            df[name]=df_.copy(deep=True)
        df=pd.concat(df,axis=1)
        return df
