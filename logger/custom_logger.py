import logging
import os
from datetime import datetime
import structlog # type: ignore

class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(),log_dir)
        os.makedirs(self.logs_dir,exist_ok=True)
        
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir,log_file)
        
    def get_logger(self,name=__file__):
        logger_name = os.path.basename(name)
        
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[file_handler,console_handler]
            )
        
        # Configure Struct Log
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to= "event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory= structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use= True
        )
        
        return structlog.get_logger(logger_name)
    
if __name__ == "__main__":
    logger = CustomLogger()
    logger = logger.get_logger(__file__)
    logger.info("Custom Logger Initiated Again")