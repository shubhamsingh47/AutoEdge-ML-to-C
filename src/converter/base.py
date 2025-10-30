from typing import Any
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

logger = CustomLogger().get_logger(__name__)

class BaseConverter:
    def __init__(self,model_path:str):
        self.model_path = model_path
        self.model: Any | None = None
        
    def load(self)->None:
        raise NotImplementedError
    
    def convert_to_c(self, *args, **kwargs)->str:
        raise NotImplementedError