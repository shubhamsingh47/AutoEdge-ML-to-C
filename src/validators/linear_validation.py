import sys
import numpy as np
from typing import Any
from src.converter.base import BaseConverter
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from src.utils import detect_linear_model_kind

logger = CustomLogger().get_logger(__name__)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def validate_linear_model_exported(model: Any, tolerance: float = 1e-6, n_samples: int = 32) -> None:
    try:
        n_features = int(model.coef_.shape[-1])
    except Exception as e:
        logger.info("Failed to get number of features: %s", e)
        raise CustomException("Failed to determine model input shape", sys)

    state = np.random.RandomState(0)
    X = state.randn(n_samples, n_features).astype(float)

    y_sklearn = model.predict(X)

    model_type = detect_linear_model_kind(model)

    # Multiclass classification
    if model_type == "classification" and model.coef_.ndim == 2 and model.coef_.shape[0] > 1:

        W = model.coef_
        b = model.intercept_

        logits = X.dot(W.T) + b

        probs_manual = softmax(logits)

        probs_sklearn = model.predict_proba(X)

        max_diff = float(np.max(np.abs(probs_sklearn - probs_manual)))
        logger.info("Multiclass softmax max prob diff: %g", max_diff)

        if max_diff > tolerance:
            raise CustomException(f"Softmax probability mismatch: max diff {max_diff:.6g}", sys)

        y_manual_classes = np.argmax(probs_manual, axis=1)
        mismatches = (y_manual_classes != y_sklearn).sum()

        if mismatches > 0:
            raise CustomException(
                f"Multiclass label mismatches {mismatches}/{n_samples}", sys
            )

        logger.info("Multiclass logistic regression validation PASSED")
        return

    # Binary Classification
    if model_type == "classification" and model.coef_.ndim == 2 and model.coef_.shape[0] == 1:
        coef = model.coef_.ravel()
        intercept = float(model.intercept_.ravel()[0])

        y_manual_logit = X.dot(coef) + intercept
        probs_manual = 1.0 / (1.0 + np.exp(-y_manual_logit))

        probs_sklearn = model.predict_proba(X)[:, 1]
        max_diff = float(np.max(np.abs(probs_sklearn - probs_manual)))

        logger.info("Binary logistic max prob diff: %g", max_diff)

        if max_diff > tolerance:
            raise CustomException(f"Binary logistic probability mismatch max diff {max_diff:.6g}", sys)

        y_manual_cls = (probs_manual >= 0.5).astype(int)
        mismatches = (y_manual_cls != y_sklearn).sum()

        if mismatches > 0:
            raise CustomException(
                f"Binary classification mismatches {mismatches}/{n_samples}", sys
            )

        logger.info("Binary logistic regression validation PASSED")
        return

    # Linear Regression
    if model_type == "regression":
        coef = model.coef_.ravel()
        intercept = float(model.intercept_.ravel()[0])
        y_manual = X.dot(coef) + intercept

        max_diff = float(np.max(np.abs(y_sklearn - y_manual)))
        logger.info("Regression max diff: %g", max_diff)

        if max_diff > tolerance:
            raise CustomException(f"Regression mismatch max diff {max_diff:.6g}", sys)

        logger.info("Linear regression validation PASSED")
        return

    raise CustomException("Unhandled model type in validation", sys)
