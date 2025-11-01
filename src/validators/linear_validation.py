import sys
import numpy as np
from typing import Any, Optional
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from src.utils import detect_linear_model_kind

logger = CustomLogger().get_logger(__name__)

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def validate_linear_model_exported(estimator: Any, scaler: Optional[Any] = None, tolerance: float = 1e-6, n_samples: int = 32) -> None:
    if estimator is None:
        raise CustomException("Validator received None estimator", sys)

    try:
        if hasattr(estimator, "coef_"):
            if estimator.coef_.ndim == 1:
                n_features = int(estimator.coef_.shape[0])
            else:
                n_features = int(estimator.coef_.shape[1])
        else:
            raise AttributeError("Estimator has no coef_ attribute")
    except Exception as e:
        logger.exception("Failed to determine number of features: %s", e)
        raise CustomException("Failed to determine model input shape", sys)

    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features).astype(float)

    # apply scaler if provided
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            logger.exception("Failed to apply scaler in validator: %s", e)
            raise CustomException("Scaler transform failed", sys)
    else:
        X_scaled = X

    try:
        y_sklearn = estimator.predict(X_scaled)
    except Exception as e:
        logger.exception("Estimator.predict failed: %s", e)
        raise CustomException("Estimator.predict failed during validation", sys)

    model_type = detect_linear_model_kind(estimator)

    # multiclass
    if model_type == "multiclass":
        W = estimator.coef_ 
        b = estimator.intercept_
        logits = X_scaled.dot(W.T) + b
        probs_manual = softmax(logits)
        try:
            probs_sklearn = estimator.predict_proba(X_scaled)
        except Exception as e:
            logger.exception("predict_proba failed: %s", e)
            raise CustomException("predict_proba missing during multiclass validation", sys)
        max_diff = float(np.max(np.abs(probs_sklearn - probs_manual)))
        logger.info("Multiclass softmax max prob diff: %g", max_diff)
        if max_diff > tolerance:
            raise CustomException(f"Softmax probability mismatch: max diff {max_diff:.6g}", sys)
        y_manual = np.argmax(probs_manual, axis=1)
        mismatches = int((y_manual != y_sklearn).sum())
        if mismatches > 0:
            raise CustomException(f"Multiclass label mismatches {mismatches}/{n_samples}", sys)
        logger.info("Multiclass validation PASSED")
        return

    # binary logistic
    if model_type == "classification":
        coef = estimator.coef_.ravel()
        intercept = float(estimator.intercept_.ravel()[0])
        logits = X_scaled.dot(coef) + intercept
        probs_manual = 1.0 / (1.0 + np.exp(-logits))
        try:
            probs_sklearn = estimator.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            logger.exception("predict_proba missing: %s", e)
            raise CustomException("predict_proba required for binary logistic validation", sys)
        max_diff = float(np.max(np.abs(probs_sklearn - probs_manual)))
        logger.info("Binary logistic max prob diff: %g", max_diff)
        if max_diff > tolerance:
            raise CustomException(f"Binary logistic probability mismatch max diff {max_diff:.6g}", sys)
        y_manual = (probs_manual >= 0.5).astype(int)
        mismatches = int((y_manual != y_sklearn).sum())
        if mismatches > 0:
            raise CustomException(f"Binary label mismatches {mismatches}/{n_samples}", sys)
        logger.info("Binary logistic validation PASSED")
        return

    # regression
    if model_type == "regression":
        coef = estimator.coef_.ravel()
        intercept = float(estimator.intercept_.ravel()[0])
        y_manual = X_scaled.dot(coef) + intercept
        max_diff = float(np.max(np.abs(y_manual - y_sklearn)))
        logger.info("Regression max diff: %g", max_diff)
        if max_diff > tolerance:
            raise CustomException(f"Regression outputs mismatch: max diff {max_diff:.6g}", sys)
        logger.info("Regression validation PASSED")
        return

    raise CustomException("Unhandled model type in validator", sys)
