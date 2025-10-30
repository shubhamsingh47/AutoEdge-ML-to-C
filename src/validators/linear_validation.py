import sys
import numpy as np
from typing import Any
from src.converter.base import BaseConverter
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from src.utils import detect_linear_model_kind

logger = CustomLogger().get_logger(__name__)

def validate_linear_model_exported(model: Any, tolerance: float = 1e-6, n_samples: int = 32) -> None:
    """
    Validate model predict() vs manual linear math used in exporter.
    For classification (LogisticRegression) we compare probabilities and also class labels if needed.
    Raises ValidationError on mismatch.
    """
    try:
        n_features = int(model.coef_.shape[-1])
    except Exception as e:
        logger.info("Failed to get number of features: %s",e)
        raise CustomException("Failed to determine model input shape",sys)
    
    state = np.random.RandomState(0)
    X = state.randn(n_samples, n_features).astype(float)

    y_sklearn = model.predict(X)

    # manual compute
    coef = np.ravel(model.coef_)
    intercept = float(np.ravel(model.intercept_)[0]) if hasattr(model, "intercept_") else 0.0
    y_manual = X.dot(coef) + intercept

    model_type = detect_linear_model_kind(model)
    if model_type == "classification":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            # manual sigmoid probabilities
            probs_manual = 1.0 / (1.0 + np.exp(-y_manual))
            max_diff = float(np.max(np.abs(probs - probs_manual)))
            logger.info("Max probability diff: %g", max_diff)
            if max_diff > tolerance:
                raise CustomException(f"Probability mismatch max diff {max_diff:.6g}",sys)

        if y_sklearn.shape == y_manual.shape:
            pass
        else:
            preds_manual = (1.0 / (1.0 + np.exp(-y_manual)) >= 0.5).astype(int)
            preds_sklearn = model.predict(X)
            mismatches = (preds_manual != preds_sklearn).sum()
            if mismatches > 0:
                raise CustomException(f"Class label mismatch count: {int(mismatches)} / {n_samples}",sys)
    else:
        # regression
        max_diff = float(np.max(np.abs(y_sklearn - y_manual)))
        logger.info("Max regression diff: %g", max_diff)
        if max_diff > tolerance:
            raise CustomException(f"Regression outputs mismatch: max diff {max_diff:.6g}",sys)

    logger.info("Linear model validation PASSED (max tolerance %g).", tolerance)