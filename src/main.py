import os
import sys
import argparse
from typing import Optional
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from src.utils import generate_unique_header_name, ensure_dir, detect_linear_model_kind
from src.converter.linear import LinearConverter
from src.validators.linear_validation import validate_linear_model_exported
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


logger = CustomLogger().get_logger(__name__)

def convert_model(model_path: str, output_dir: str = "./generated", validate: bool = True) -> str:
    logger.info("Starting conversion: %s", model_path)

    try:
        converter = LinearConverter(model_path)
        converter.load()

        model_type = detect_linear_model_kind(converter.model)
        logger.info("Auto-detected model type: %s", model_type)

        c_code = converter.convert_to_c(func_name="predict_model")

        ensure_dir(output_dir)
        file_name = generate_unique_header_name(converter.model, model_path)
        out_path = os.path.join(output_dir, file_name)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(c_code)

        logger.info("Saved C header: %s", out_path)

        if validate:
            logger.info("Running Python-only validation…")
            validate_linear_model_exported(estimator=converter.model, scaler=converter.scaler)
            logger.info("Validation passed.")

        return out_path

    except Exception as e:
        logger.exception("Conversion failed")
        raise CustomException(f"Model conversion failed: {e}", sys)

def cli_entry(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog="model2c", description="Convert sklearn linear/pipeline .pkl into C header (.h)")
    parser.add_argument("--model", "-m", required=True, help="Path to model .pkl")
    parser.add_argument("--out", "-o", default="./generated", help="Output directory")
    parser.add_argument("--no-validate", action="store_true", help="Skip python-side validation")
    args = parser.parse_args(argv)

    try:
        output_file = convert_model(args.model, args.out, validate=not args.no_validate)
        print(f"Conversion successful → {output_file}")
    except Exception as e:
        logger.exception("Conversion failed")
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    cli_entry()
