import os
import hashlib
import datetime
from typing import Any, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,LogisticRegression

def detect_linear_model_kind(model: Any) -> str:
    if isinstance(model, LogisticRegression):
        n_classes = getattr(model, "classes_", None)
        if n_classes is not None and getattr(n_classes, "shape", (1,))[0] > 2:
            return "multiclass"
        return "classification"

    if isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet)):
        return "regression"

    if isinstance(model, RegressorMixin):
        return "regression"
    if isinstance(model, ClassifierMixin):
        return "classification"

    raise TypeError(f"Unsupported linear model type: {type(model)}")


def extract_pipeline_components(obj: Any) -> Tuple[Optional[Any], Any]:
    if isinstance(obj, Pipeline):
        scaler = None
        estimator = None
        for name, step in obj.steps:
            # detect scaler types we support
            if isinstance(step, (StandardScaler, MinMaxScaler)):
                scaler = step
            else:
                estimator = step
        if estimator is None:
            raise ValueError("Pipeline does not contain an estimator at the end.")
        return scaler, estimator
    return None, obj

unwrap_pipeline = extract_pipeline_components


def is_standard_scaler(obj: Any) -> bool:
    from sklearn.preprocessing import StandardScaler
    return isinstance(obj, StandardScaler)

def is_minmax_scaler(obj: Any) -> bool:
    from sklearn.preprocessing import MinMaxScaler
    return isinstance(obj, MinMaxScaler)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def generate_unique_header_name(model: Any, model_path: str) -> str:
    class_name = model.__class__.__name__
    try:
        model_type = detect_linear_model_kind(model)
    except Exception:
        model_type = "unknown"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:6]
    else:
        file_hash = hashlib.md5(model_path.encode()).hexdigest()[:6]

    return f"{class_name}_{model_type}_{timestamp}_{file_hash}.h"
