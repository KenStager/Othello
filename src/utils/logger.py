import os, time, sys

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

class TSVLogger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write("time	key	value\n")
    def log(self, key, value):
        with open(self.path, 'a') as f:
            f.write(f"{int(time.time())}\t{key}\t{value}\n")
