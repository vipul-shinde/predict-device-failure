import logging

from failure_prediction_model.config import config

logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())


with open(config.PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
