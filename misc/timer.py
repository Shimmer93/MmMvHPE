import time
from contextlib import contextmanager
from functools import wraps

@contextmanager
def timer(name="Operation"):
    """Context manager for timing code blocks.
    
    Usage:
        with timer("Data loading"):
            load_data()
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.4f} seconds")

def time_function(func):
    """Decorator for timing functions.
    
    Usage:
        @time_function
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__}: {elapsed:.4f} seconds")
        return result
    return wrapper

class Timer:
    """Class-based timer for tracking multiple steps.
    
    Usage:
        timer = Timer()
        timer.start("data_loading")
        load_data()
        timer.stop("data_loading")
        timer.report()
    """
    def __init__(self):
        self.times = {}
        self.starts = {}
    
    def start(self, name):
        self.starts[name] = time.time()
    
    def stop(self, name):
        if name in self.starts:
            elapsed = time.time() - self.starts[name]
            self.times[name] = self.times.get(name, 0) + elapsed
            del self.starts[name]
    
    def report(self):
        print("\n=== Timing Report ===")
        total = sum(self.times.values())
        for name, elapsed in sorted(self.times.items(), key=lambda x: -x[1]):
            percentage = (elapsed / total * 100) if total > 0 else 0
            print(f"{name}: {elapsed:.4f}s ({percentage:.1f}%)")
        print(f"Total: {total:.4f}s")
