import os
import logging
from typing import List


class ExperimentLogger:

    def __init__(self, logger_name: str, log_dir: str, field_names: List[str]):
        self.logger = self._configure_logger(logger_name, log_dir)
        self.csv_logger = self._configure_csv_logger(logger_name, log_dir)
        self.val_csv_logger = self._configure_csv_logger(f'val_{logger_name}', log_dir)
        self.stderr_fmt = self._get_stderr_fmt(field_names)
        self.csv_fmt = self._get_csv_fmt(field_names)

    @staticmethod
    def _configure_logger(logger_name: str, log_dir: str):
        level = logging.INFO
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s|%(message)s')

        log_file = os.path.join(log_dir, f'{logger_name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.info(f'Logging to {log_file}')
        return logger

    @staticmethod
    def _configure_csv_logger(logger_name: str, log_dir: str):
        level = logging.INFO
        logger = logging.getLogger(f'{logger_name}.csv')
        logger.propagate = False
        logger.setLevel(level)
        formatter = logging.Formatter('%(message)s')

        log_file = os.path.join(log_dir, f'{logger_name}.csv')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def _get_stderr_fmt(self, field_names: List[str]):
        log_header = '|'.join([f'{field_name:^12}' for field_name in field_names])
        log_format = '|'.join([f'{{{field_name}:^12.4f}}' for field_name in field_names])
        self.logger.info(log_header)
        return log_format

    def _get_csv_fmt(self, field_names: List[str]):
        log_header = ','.join([f'{field_name}' for field_name in field_names])
        log_format = ','.join([f'{{{field_name}:.4f}}' for field_name in field_names])
        self.csv_logger.info(log_header)
        return log_format

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def log(self, *args, **kwargs):
        self.logger.info(self.stderr_fmt.format(*args, **kwargs))
        self.csv_logger.info(self.csv_fmt.format(*args, **kwargs))

    def log_val_metrics(self, *args, **kwargs):
        self.val_csv_logger.info(*args, **kwargs)


def configure_logger(logger_name: str, logger_level=logging.INFO, log_dir: str = '/tmp'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    csv_formatter = logging.Formatter('%(asctime)s|%(message)s')

    log_file = os.path.join(log_dir, f'{logger_name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(csv_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger_level)
    stream_handler.setFormatter(csv_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(f'Logging to {log_file}')

    return logger