import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly


class Winsorizer(BaseEstimator, TransformerMixin):
    # basically says "when creating a winsorizer, provide a lower and upper quantile"
    def __init__(self, lower_quantile, upper_quantile):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        # added axis=0 to calculate quantiles for each column
        return self  # means: "return this winsorizer, which now remembers the calculated limits"

    def transform(self, X):
        # check that fit has created both limits -> impoted at top of file
        check_is_fitted(self, ["lower_quantile_", "upper_quantile_"])
        # return the data where np.clip caps each column's values at its learned lower/upper limits
        return np.clip(X, self.lower_quantile_, self.upper_quantile_)
