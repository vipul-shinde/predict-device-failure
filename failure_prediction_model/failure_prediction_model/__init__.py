import logging

from failure_prediction_model.config import config

logging.getLogger(config.PACKAGE_NAME).addHandler(logging.NullHandler())


with open(config.PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
