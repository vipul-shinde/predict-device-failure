# Importing the libraries
import logging
import joblib

import pandas as pd
from failure_prediction_model import __version__ as _version
from failure_prediction_model.config import config
from failure_prediction_model.processing.data_manager import load_pipeline, load_dataset

logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
failure_detection_pipeline = load_pipeline(file_name=pipeline_file_name)


def make_predictions(*, input_data, save_as: str) -> dict:
    """Predict probability of device failure for the test data

    Parameters
    ----------
    input_data : _type_
        A csv file containing the test data
    save_as : str, optional
        name to save the file as, by default None

    Returns
    -------
    dict
        predicted probability values
    """

    test_data = load_dataset(file_name=input_data, train=False)

    predictions = failure_detection_pipeline.predict_proba(
        test_data[config.TEST_FEATURES])

    logger.info(
        f"Predictions: {predictions}"
    )

    # Append predicted probability to the test dataset
    res = pd.DataFrame(predictions)
    res.index = test_data[config.TEST_FEATURES].index
    res.columns = ["failure_probability"]
    final_out = pd.concat(test_data, res, axis=1)
    final_out.to_csv(f"{config.OUTPUT_DIR}/{save_as}", index=False)

    results = {"predictions": predictions.tolist()}
    return results
