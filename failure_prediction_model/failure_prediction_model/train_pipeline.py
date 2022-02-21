# importing the libraries
import logging

from failure_prediction_model.config import config
from failure_prediction_model import __version__ as _version

from failure_prediction_model.processing.features import sampling_pipeline
from failure_prediction_model.processing.data_manager import load_dataset, save_pipeline
from failure_prediction_model.pipeline import vc_pipeline
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model
    """

    # Load the training dataset
    data = load_dataset(file_name=config.TRAIN_DATA,
                        save=True, save_as=config.INTERIM_TRAIN_DATA)

    X = data.drop("failure", axis=1)
    y = data.loc[:, "failure"]

    # Splitting the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, shuffle=True, stratify=y)

    # Resampling the training dataset to tackle imbalance problem
    X_train_rs, y_train_rs = sampling_pipeline.fit_resample(X_train, y_train)

    # Training the model
    vc_pipeline.fit(X_train_rs, y_train_rs)

    # Save the trained model
    logger.warning(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=vc_pipeline)


if __name__ == "__main__":
    run_training()
