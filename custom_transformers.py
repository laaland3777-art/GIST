import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class UlcerBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping_ = None

    def fit(self, X, y=None):
        s = pd.Series(X.squeeze()).astype(str).str.strip()
        unique_vals = list(pd.Series(s).dropna().unique())

        positive_candidates = {'1', 'yes', 'y', 'true', 'positive', '有', '是', '阳性'}
        negative_candidates = {'0', 'no', 'n', 'false', 'negative', '无', '否', '阴性'}

        if len(unique_vals) == 2:
            pos_found = [v for v in unique_vals if v.lower() in positive_candidates]
            neg_found = [v for v in unique_vals if v.lower() in negative_candidates]

            if len(pos_found) == 1 and len(neg_found) == 1:
                self.mapping_ = {
                    neg_found[0]: 0,
                    pos_found[0]: 1
                }
            else:
                sorted_vals = sorted(unique_vals)
                self.mapping_ = {
                    sorted_vals[0]: 0,
                    sorted_vals[1]: 1
                }
        else:
            raise ValueError(f"Ulcer 列不是二分类变量，检测到类别: {unique_vals}")

        return self

    def transform(self, X):
        s = pd.Series(X.squeeze()).astype(str).str.strip()
        encoded = s.map(self.mapping_)
        if encoded.isnull().any():
            unseen = s[encoded.isnull()].unique().tolist()
            raise ValueError(f"Ulcer 列出现训练阶段未见过的新类别: {unseen}")
        return encoded.astype(int).to_numpy().reshape(-1, 1)
