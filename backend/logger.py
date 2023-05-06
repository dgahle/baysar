########################################################################################################################
### A module to set up and configure the logger for this model #########################################################
########################################################################################################################
#
# Imports
import logging
from datetime import datetime
from json import load
from pathlib import Path

from tqdm import tqdm

# Variables
# Load config
REPO_PATH: Path = Path(__file__).parent.parent
CONFIG_PATH: Path = REPO_PATH / "config.json"
with open(CONFIG_PATH, "r") as f:
    CONFIG_DICT = load(f)
# Paths
OUTPUT_PATH: Path = REPO_PATH / "output"
# Logger
logger_datefmt = "%Y-%m-%d %H:%M:%S"
log_msg_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger_format_dict = {"format": log_msg_format, "datefmt": logger_datefmt}
datetime_datefmt = "%Y%m%d_%H%M%S"
timestamp = datetime.now().strftime(datetime_datefmt)
logger_path = OUTPUT_PATH / f"{CONFIG_DICT['report_name']}_{timestamp}.log"


# Functions and classes
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(
    name: str, set_cmd: bool = True, set_path: bool = True
) -> logging.Logger:
    """
    Set up the logging module for the model run.

    :param (str) name:
        Name of the logger object returned and used in the log to note which loggers are writing to the log.
    :param (bool) set_cmd: True
        Switch for writing the log to the stdout
    :param (bool) set_path: True
        Switch for writing the log to file (*.log)
    :return:
    """
    # Instantiate the logger
    logger_name = Path(__file__).name if name is None else name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Writing the logger to file
    filename = logger_path if set_path else None
    logging.basicConfig(filename=filename, level=logging.INFO, **logger_format_dict)
    # Printing the logger to the command line
    set_cmd_check: bool = set_path is False and set_cmd is True
    set_cmd: bool = False if set_cmd_check else set_cmd
    if set_cmd:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter(fmt=log_msg_format, datefmt=logger_datefmt)
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
    # # Add tqdm progress bar compatibility
    # logger.addHandler(TqdmLoggingHandler())

    return logger


def main() -> None:
    filename: str = Path(__file__).name
    logger = get_logger(filename, set_path=False)
    log_msg: str = f"Running main()"
    logger.info(log_msg)
    logger.debug(log_msg)
    logger.warning(log_msg)
    logger.error(log_msg)
    logger.critical(log_msg)
    pass


if __name__ == "__main__":
    main()
