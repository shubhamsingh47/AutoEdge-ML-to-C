import tempfile
import joblib
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge, LogisticRegression
from src.main import convert_model
from src.utils import load_model

def test_regression_conversion(tmp_path):
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=0)
    m = Ridge().fit(X, y)
    p = tmp_path / "reg.pkl"
    joblib.dump(m, str(p))

    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    header = convert_model(str(p), output_dir=output_dir, validate=True)
    assert header.endswith("model.h")

def test_classification_conversion(tmp_path):
    X, y = make_classification(n_features=6, n_informative=3, n_redundant=2, n_repeated=1)
    m = LogisticRegression(max_iter=1000).fit(X, y)
    p = tmp_path / "clf.pkl"
    joblib.dump(m, str(p))

    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    header = convert_model(str(p), output_dir=output_dir, validate=True)
    assert header.endswith("model.h")
