import os, time

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def timer(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"⏱ {func.__name__} took {time.time() - start:.2f}s")
        return res
    return wrapper