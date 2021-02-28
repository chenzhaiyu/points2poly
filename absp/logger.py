import logging
import numpy as np


def attach_to_log(level=logging.INFO,
                  filepath=None,
                  colors=True,
                  capture_warnings=True):
    """
    Attach a stream handler to all loggers.

    Parameters
    ------------
    level : enum (int)
      Logging level, like logging.INFO
    colors : bool
      If True try to use colorlog formatter
    capture_warnings: bool
      If True capture warnings
    filepath: None or str
    path to save the logfile
    """

    # make sure we log warnings from the warnings module
    logging.captureWarnings(capture_warnings)

    # create a basic formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
        "%Y-%m-%d %H:%M:%S")
    if colors:
        try:
            from colorlog import ColoredFormatter
            formatter = ColoredFormatter(
                ("%(log_color)s%(levelname)-8s%(reset)s " +
                 "%(filename)17s:%(lineno)-4s  %(blue)4s%(message)s"),
                datefmt=None,
                reset=True,
                log_colors={'DEBUG': 'cyan',
                            'INFO': 'green',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'red'})
        except ImportError:
            pass

    # if no handler was passed use a StreamHandler
    logger = logging.getLogger()
    logger.setLevel(level)

    if not any([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if filepath and not any([isinstance(handler, logging.FileHandler) for handler in logger.handlers]):
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # set nicer numpy print options
    np.set_printoptions(precision=5, suppress=True)

    return logger
