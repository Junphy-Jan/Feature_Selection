# coding:utf-8
import logging
import os
from logging import handlers


def get_logger(log_filename, level=logging.DEBUG, when='midnight', back_count=0):
    logger = logging.getLogger(log_filename)
    logger.setLevel(level)
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__))) + os.sep + "logs" + os.sep
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, log_filename)
    formatter = logging.Formatter('%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s')
    if not logger.handlers:
        fh = logging.handlers.TimedRotatingFileHandler(
            filename=log_file_path,
            when=when,
            backupCount=back_count,
            encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


if __name__ == "__main__":
    """
    logger = get_logger("my.log")
    logger.debug("debug test")
    logger.info("info test")
    logger.warn("warn test")
    logger.error("error test")"""
