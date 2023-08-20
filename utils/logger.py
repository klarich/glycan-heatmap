import logging


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create a basic logger
    :param name: Name for the logger
    :param level: logging level
    :return: an instance of logging.Logger
    """
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d:%H:%M:%S"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
