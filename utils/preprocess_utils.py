import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

def boxcox_on_skewed_features(df, target, s: float):
    """
    Get a df, target and s (skewness), returns boxcox on df.
    """

    num_features = list(df.select_dtypes(include=np.number).columns)
    num_features.remove(target)

    skewed_features = df[num_features].apply(lambda x: skew(x))
    high_skew = skewed_features[abs(skewed_features) > s]

    for feat in high_skew.index:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))


def logtransform_on_skewed_features(df, target, s: float):
    """
    Get a df, target and s (skewness), returns log1p on df.
    """

    num_features = list(df_all.select_dtypes(include=np.number).columns)
    num_features.remove(target)

    skewed_features = df[num_features].apply(lambda x: skew(x))
    high_skew = skewed_features[abs(skewed_features) > s]

    df[high_skew] = np.log1p(df[high_skew])
