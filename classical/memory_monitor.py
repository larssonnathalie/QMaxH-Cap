# memory_monitor.py

import psutil
import os
import threading
import time

class MemoryTracker:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.running = False
        self.peak_memory = 0.0
        self.thread = None

    def _track(self):
        process = psutil.Process(os.getpid())
        while self.running:
            mem = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB
            if mem > self.peak_memory:
                self.peak_memory = mem
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_peak(self):
        return self.peak_memory

def get_memory_usage():
    """
    Returns the current memory usage (in MB).
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2
