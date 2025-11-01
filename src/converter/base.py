from typing import Any, Optional
import joblib
from logger.custom_logger import CustomLogger
from src.utils import unwrap_pipeline

logger = CustomLogger().get_logger(__name__)

class BaseConverter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.raw_model: Optional[Any] = None
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None

    def load(self, model_obj: Optional[Any] = None) -> None:
        if model_obj is not None:
            self.raw_model = model_obj
            logger.info("BaseConverter.load(): using provided model object")
        else:
            if not self.model_path:
                raise ValueError("No model_path provided to BaseConverter")
            try:
                self.raw_model = joblib.load(self.model_path)
                logger.info("BaseConverter.load(): loaded from %s", self.model_path)
            except Exception as e:
                logger.exception("BaseConverter.load(): joblib.load failed: %s", e)
                raise

        try:
            scaler, estimator = unwrap_pipeline(self.raw_model)
            self.scaler = scaler
            self.model = estimator
            logger.info("BaseConverter.load(): extracted estimator=%s scaler=%s",
                        type(self.model).__name__, type(self.scaler).__name__ if self.scaler else None)
            return
        except Exception:
            pass

        if isinstance(self.raw_model, dict):
            if "model" not in self.raw_model:
                raise ValueError("Saved dict must contain key 'model'")
            self.model = self.raw_model["model"]
            self.scaler = self.raw_model.get("scaler", None)
            logger.info("BaseConverter.load(): loaded dict -> estimator=%s scaler=%s",
                        type(self.model).__name__, type(self.scaler).__name__ if self.scaler else None)
            return

        # otherwise - raw model as final
        self.model = self.raw_model
        self.scaler = None
        logger.info("BaseConverter.load(): final estimator=%s", type(self.model).__name__)
