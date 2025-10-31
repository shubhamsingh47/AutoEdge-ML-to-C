import numpy as np
from src.utils import load_model
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BaseConverter(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_multiclass = False

    def load(self):
        self.model = load_model(self.model_path)
        self._extract_pipeline()
        
    def _extract_pipeline(self):
        if isinstance(self.model, Pipeline):
            for step_name, step in self.model.steps:
                if isinstance(step, (StandardScaler, MinMaxScaler)):
                    self.scaler = step
                else:
                    self.model = step

        # multiclass
        if hasattr(self.model, "coef_") and len(self.model.coef_.shape) > 1:
            self.is_multiclass = True

    @abstractmethod
    def convert_to_c(self, func_name: str) -> str:
        pass
