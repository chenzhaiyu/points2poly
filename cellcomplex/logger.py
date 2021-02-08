import logging
import trimesh


def get_logger(filepath=None, level=logging.INFO):
    """
    :param level: logging level. defaults to logging.INFO.
    :param filepath: path to save the log.
    :return: logger object.
    """

    def add_file_handler(logger_without_file_handler):
        logfile = logging.FileHandler(filepath)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
        logfile.setFormatter(formatter)
        logger_without_file_handler.addHandler(logfile)

    logger = logging.getLogger('trimesh')

    if not any([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]):
        trimesh.util.attach_to_log(level=level)

    if filepath and not any([isinstance(handler, logging.FileHandler) for handler in logger.handlers]):
        add_file_handler(logger)

    return logger
