import os
import re
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
    return isinstance(obj, StandardScaler)

def is_minmax_scaler(obj: Any) -> bool:
    return isinstance(obj, MinMaxScaler)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def detect_scaler_in_pipeline(pipeline):
    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            if isinstance(step, StandardScaler):
                return "stdscaler"
            if isinstance(step, MinMaxScaler):
                return "minmax"
    return "none"


def determine_model_type(model):
    if isinstance(model, LogisticRegression):
        if model.multi_class == "multinomial" or model.classes_.shape[0] > 2:
            return "classification_multiclass"
        
        return "classification_binary"

    if isinstance(model, RegressorMixin):
        return "regression"

    if isinstance(model, ClassifierMixin):
        return "classification_binary"

    return "other"


def generate_clean_header_name(model, raw_model_obj, model_path):
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    base_name = re.sub(r"[^a-zA-Z0-9_]", "_", base_name).lower()

    model_type = determine_model_type(model)
    scaler_flag = detect_scaler_in_pipeline(raw_model_obj)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_name = f"{base_name}__{model_type}__{scaler_flag}__{timestamp}.h"

    return file_name
