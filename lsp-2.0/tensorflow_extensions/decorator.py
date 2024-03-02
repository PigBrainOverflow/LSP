from typing import Tuple, Dict

# diy block
def block_assert(
    *,
    inputs: Tuple[Dict],
    output: Dict
):
    # return the original function
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# tensor creation
def tensor_assert(
    *,
    output: Dict
):
    # return the original function
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# conv2d layer
def conv2d_assert(
    *,
    filters: int, kernel_size: int, strides: int, padding: str
):
    # return the original function
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# pooling2d layer
def pooling2d_assert(
    *,
    pool_size: int, strides: int, padding: str
):
    # return the original function
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# scalar operation (e.g. relu)
def scalar_op_assert(func):
    # return the original function
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# element-wise operation (e.g. +, -, *, /)
def ewise_op_assert(func):
    # return the original function
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper