import time


class Timer:

    def __init__(self):
        self.time = 0

    def __enter__(self):
        self.time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.time
        print(f'took {elapsed:.5f} s')
