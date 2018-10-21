import datetime

SUBMISSION_DIR = './submissions'
DATA_DIR = '../data'
SNAPSHOT_DIR = './snapshots'
LOG_DIR = './logs'


def get_current_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')