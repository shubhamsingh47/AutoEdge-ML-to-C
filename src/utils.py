import os
import sys
import joblib
import hashlib
import numpy as np
from typing import Any
from datetime import datetime
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from sklearn.base import RegressorMixin, ClassifierMixin

logger = CustomLogger().get_logger(__name__)

def load_model(path: str) -> Any:
    if not os.path.exists(path):
        logger.error("Model file not found")
        try:
            raise FileNotFoundError("Model not found")
        except Exception as e:
            raise CustomException(str(e), sys)
    try:
        model = joblib.load(path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.info('Failed to load model')
        raise CustomException(str(e),sys) 
        
    
def detect_linear_model_kind(model:Any)->str:
    cls = model.__class__.__name__
    cls_lower = cls.lower()
    linear_names = ("linearregression","ridge","lasso","elasticnet")
    logistic_names = ("logisticregression",)
    if any(n in cls_lower for n in linear_names):
        return "regression"
    if any(n in cls_lower for n in logistic_names):
        return "classification"    
    if hasattr(model,'coef_'):
        return "regression"
    try:
        raise ModuleNotFoundError("Model type not detected ")
    except Exception as e:
        raise CustomException(str(e), sys)


def ensure_dir(path:str)->None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name,exist_ok=True)
        

def generate_unique_header_name(model, model_path: str) -> str:
    try:
        class_name = model.__class__.__name__

        if isinstance(model, RegressorMixin):
            model_type = "regression"
        elif isinstance(model, ClassifierMixin):
            model_type = "classification"
        else:
            model_type = "other"

        with open(model_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:6]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"{class_name}_{model_type}_{timestamp}_{file_hash}.h"
    
    except Exception as e:
        raise CustomException("Failed to generate unique header.")
