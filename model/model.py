import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def read_raw(test_data_fname="data/test_data.csv",
             user_data_fname="data/viewer_data.csv",
             use_medians=True):
    test_data = pd.read_csv(test_data_fname)
    user_data = pd.read_csv(user_data_fname)
    print("The size of the original dataset is {}".format(len(test_data)))

    merged = test_data.merge(user_data,
                             left_on='viewer_id',
                             right_on='viewer_id',
                             how='outer')

    print("The size of the dataset after merge is {}".format(len(merged)))

    # As the first iteration join all unknown providers
    merged["tv_provider"] = merged["tv_provider"].fillna("Unknown")
    merged["city"] = merged["city"].fillna("Unknown")
    merged["gender"] = merged["gender"].fillna("Unknown")
    merged["date"] = pd.to_datetime(merged["date"])
    if use_medians:
        merged["age"] = merged["age"].fillna(merged["age"].median())
    return merged


def read_dataset(test_data_fname="data/test_data.csv",
                 user_data_fname="data/viewer_data.csv"):
    raw = read_raw(test_data_fname, user_data_fname)
    data = raw.drop(columns="watched")
    X, y = RandomUnderSampler().fit_resample(data, raw["watched"])
    return train_test_split(pd.DataFrame(X, columns=data.columns), y)


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, records=False, dtype=None):
        self.columns = columns
        self.records = records
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.dtype is not None:
            X[self.columns] = X[self.columns].astype(self.dtype)
        if self.records:
            return X[self.columns].to_dict(orient="records")
        return X[self.columns]


class PandasFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, iname, oname, func):
        self.iname = iname
        self.oname = oname
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = pd.DataFrame(index=X.index)
        output[self.oname] = X[self.iname].apply(self.func)
        return output


def categorical(columns):
    return make_pipeline(
        PandasSelector(columns, records=True),
        DictVectorizer(),
    )


def numeric(columns, dtype):
    return make_pipeline(
        PandasSelector(columns, dtype=dtype),
        StandardScaler()
    )


def build_model():
    model = make_pipeline(
        make_union(
            categorical(["tv_make", "tv_provider", "gender", "city"]),
            categorical(["test"]),  # NB: keep it separate
            numeric(["tv_size", "total_time_watched", "age"], dtype=float),
            PandasSelector(["uhd_capable"], dtype=int),
            make_pipeline(
                PandasFuncTransformer("date", "weekdays",
                                      lambda x: x.weekday()),
                categorical(["weekdays"])
            )
        ),
        RandomForestClassifier()
    )
    return model
