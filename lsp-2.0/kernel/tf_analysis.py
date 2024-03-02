from typing import List
from .core import *
import ast
from .tensorflow_extensions import *
from copy import copy

scalar_binops = (
    ast.Add, ast.Sub, ast.Mult, ast.Div
)

matmul_binop = ast.MatMult

decorators_with_no_args = {
    "scalar_op_assert": scalar_op_assert,
    "ewise_op_assert": ewise_op_assert
}

decorators_with_args = {
    "block_assert": block_assert,
    "tensor_assert": tensor_assert,
    "conv2d_assert": conv2d_assert,
    "pooling2d_assert": pooling2d_assert
}

class Analyzer:
    _diag_msgs: List

#################################################
##### utility methods
#################################################
    @staticmethod
    def calculate_conv2d_shape(input_shape: Tuple[int], filters: int, kernel_size: int, strides: int, padding: str) -> Tuple[int] | None:
        # return the shape of the result of conv2d
        # None if the input shape is not compatible
        if padding == "valid":
            h = (input_shape[1] - kernel_size) // strides + 1
            w = (input_shape[2] - kernel_size) // strides + 1
        elif padding == "same":
            h = input_shape[1] // strides
            w = input_shape[2] // strides
        else:
            return None
        return (input_shape[0], h, w, filters)

    @staticmethod
    def calculate_pooling2d_shape(input_shape: Tuple[int], pool_size: int, strides: int, padding: str) -> Tuple[int] | None:
        # return the shape of the result of pooling2d
        # None if the input shape is not compatible
        if padding == "valid":
            h = (input_shape[1] - pool_size) // strides + 1
            w = (input_shape[2] - pool_size) // strides + 1
        elif padding == "same":
            h = input_shape[1] // strides
            w = input_shape[2] // strides
        else:
            return None
        return (input_shape[0], h, w, input_shape[3])

    @staticmethod
    def calculate_matmul_shape(shape1: Tuple[int], shape2: Tuple[int]) -> Tuple[int] | None:
        # return the shape of the result of matrix multiplication
        # None if the shapes are not compatible
        if shape1[:-2] != shape2[:-2]:  # other dimensions except the last two should be the same
            return None

        last_dim1 = shape1[-2:]
        last_dim2 = shape2[-2:]
        if len(last_dim1) != 2 or len(last_dim2) != 2:
            return None
        if last_dim1[1] != last_dim2[0]:
            return None
        result_shape = shape1[:-1] + (last_dim2[1],)
        return result_shape

    def __init__(self):
        self._diag_msgs = []

    def add_diag_msg(self, msg: str, **info):
        self._diag_msgs.append(
            {
                **info,
                "message": msg
            }
        )

#################################################
##### check function call
#################################################
    def check_block_function_call(self, func: BasicBlock, args: List, **info) -> Tensor | None:
        func_inputs: Tuple[Dict[str, Any]] = func.kwargs["inputs"]
        if len(func_inputs) != len(args):
            self.add_diag_msg("number of arguments does not match", **info, expected=len(func_inputs), actual=len(args))
        elif any(not isinstance(arg, Tensor) for arg in args):
            self.add_diag_msg("arguments should be tensors", **info)
        elif any(arg.shape != input["shape"] for arg, input in zip(args, func_inputs)):
            self.add_diag_msg("arguments should have compatible shapes", **info, expected=[input["shape"] for input in func_inputs], actual=[arg.shape for arg in args])
        elif any(arg.dtype != input["dtype"] for arg, input in zip(args, func_inputs)):
            self.add_diag_msg("arguments should have the same dtype", **info, expected=[input["dtype"] for input in func_inputs], actual=[arg.dtype for arg in args])
        elif any(arg.device != input["device"] for arg, input in zip(args, func_inputs)):
            self.add_diag_msg("arguments should have the same device", **info, expected=[input["device"] for input in func_inputs], actual=[arg.device for arg in args])
        else:
            return Tensor(**func.kwargs["output"])
        return None

    def check_conv2d_function_call(self, func: BasicBlock, args: List, **info) -> Tensor | None:
        if len(args) != 1:
            self.add_diag_msg("conv2d function requires one argument", **info, actual=len(args))
        elif not isinstance(args[0], Tensor):
            self.add_diag_msg("conv2d function requires a tensor argument", **info)
        elif len(args[0].shape) != 4:
            self.add_diag_msg("conv2d function requires a 4D tensor argument", **info, actual=args[0].shape)
        else:
            ret_shape = Analyzer.calculate_conv2d_shape(args[0].shape, **func.kwargs)
            if ret_shape is None:
                self.add_diag_msg("conv2d function requires compatible input shape", **info, input_shape=args[0].shape)
            return Tensor(shape=ret_shape, dtype=args[0].dtype, device=args[0].device)
        return None

    def check_pooling2d_function_call(self, func: BasicBlock, args: List, **info) -> Tensor | None:
        if len(args) != 1:
            self.add_diag_msg("pooling2d function requires one argument", **info, actual=len(args))
        elif not isinstance(args[0], Tensor):
            self.add_diag_msg("pooling2d function requires a tensor argument", **info)
        elif len(args[0].shape) != 4:
            self.add_diag_msg("pooling2d function requires a 4D tensor argument", **info, actual=args[0].shape)
        else:
            ret_shape = Analyzer.calculate_pooling2d_shape(args[0].shape, **func.kwargs)
            if ret_shape is None:
                self.add_diag_msg("pooling2d function requires compatible input shape", **info, input_shape=args[0].shape)
            return Tensor(shape=ret_shape, dtype=args[0].dtype, device=args[0].device)
        return None

#################################################
##### check assertions
#################################################
    def check_block_assert(self, kwargs: Dict[str, Any] | None, **info) -> BasicBlock | None:
        if kwargs is None:
            self.add_diag_msg("block_assert requires keyword arguments", **info)
            return None
        if "inputs" not in kwargs or "output" not in kwargs:
            self.add_diag_msg("block_assert requires inputs and output", **info)
            return None
        inputs = kwargs["inputs"]
        output = kwargs["output"]
        if not isinstance(inputs, tuple) or not isinstance(output, dict):
            self.add_diag_msg("block_assert requires inputs to be a tuple and output to be a dictionary", **info)
            return None
        for i, inp in enumerate(inputs):
            if not isinstance(inp, dict):
                self.add_diag_msg(f"input {i} is not a dictionary", **info)
                return None
            if "shape" not in inp or "dtype" not in inp or "device" not in inp:
                self.add_diag_msg(f"input {i} does not have shape, dtype and device", **info)
                return None
        if "shape" not in output or "dtype" not in output or "device" not in output:
            self.add_diag_msg("output does not have shape, dtype and device", **info)
            return None
        return BasicBlock("block_assert", kwargs)

    def check_tensor_assert(self, kwargs: Dict[str, Any] | None, **info) -> BasicBlock | None:
        if kwargs is None:
            self.add_diag_msg("tensor_assert requires keyword arguments", **info)
            return None
        output = kwargs.get("output", None)
        if "shape" not in output or "dtype" not in output or "device" not in output:
            self.add_diag_msg("tensor_assert requires shape, dtype and device", **info)
            return None
        return BasicBlock("tensor_assert", kwargs)

    def check_conv2d_assert(self, kwargs: Dict[str, Any] | None, **info) -> BasicBlock | None:
        if kwargs is None:
            self.add_diag_msg("conv2d_assert requires keyword arguments", **info)
            return None
        if "filters" not in kwargs or "kernel_size" not in kwargs or "strides" not in kwargs or "padding" not in kwargs:
            self.add_diag_msg("conv2d_assert requires filters, kernel_size, strides and padding", **info)
            return None
        return BasicBlock("conv2d_assert", kwargs)

    def check_pooling2d_assert(self, kwargs: Dict[str, Any] | None, **info) -> BasicBlock | None:
        if kwargs is None:
            self.add_diag_msg("pooling2d_assert requires keyword arguments", **info)
            return None
        if "pool_size" not in kwargs or "strides" not in kwargs or "padding" not in kwargs:
            self.add_diag_msg("pooling2d_assert requires pool_size, strides and padding", **info)
            return None
        return BasicBlock("pooling2d_assert", kwargs)

#################################################
##### properties
#################################################
    @property
    def diag_msgs(self) -> List:
        return self._diag_msgs

#################################################
##### analysis methods
#################################################
    def analyze_decorator_with_no_args(self, decorator_id: str) -> BasicBlock | None:
        if decorator_id in decorators_with_no_args:
            return BasicBlock(decorator_id)
        return None

    def analyze_decorator_with_args(self, decorator_id: str, keywords: List[ast.keyword], **info) -> BasicBlock | None:
        if decorator_id in decorators_with_args:
            try:
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in keywords}
            except ValueError:
                self.add_diag_msg("invalid keyword argument in decorator, must be constants", **info)
                return None
            # check if the kwargs are valid
            if decorator_id == "block_assert":
                return self.check_block_assert(kwargs, **info)
            elif decorator_id == "tensor_assert":
                return self.check_tensor_assert(kwargs, **info)
            elif decorator_id == "conv2d_assert":
                return self.check_conv2d_assert(kwargs, **info)
            elif decorator_id == "pooling2d_assert":
                return self.check_pooling2d_assert(kwargs, **info)
            # to be implemented more
        return None

    def analyze_call(self, local_ids: Dict[str, Variable], func_id: str, args: List, **info) -> Tensor | None:
        if func_id in local_ids:
            func = local_ids[func_id]
            if isinstance(func, BasicBlock):
                if func.name in decorators_with_no_args:
                    if func.name == "scalar_op_assert" or func.name == "ewise_op_assert":
                        return copy(args[0])
                    # to be implemented more
                    return None
                elif func.name in decorators_with_args:
                    if func.name == "block_assert":
                        return self.check_block_function_call(func, args, **info)
                    if func.name == "tensor_assert":
                        return Tensor(**func.kwargs["output"])
                    if func.name == "conv2d_assert":
                        return self.check_conv2d_function_call(func, args, **info)
                    if func.name == "pooling2d_assert":
                        return self.check_pooling2d_function_call(func, args, **info)
                    # to be implemented more
                    return None
        return None

    def analyze_scalar_binop(self, op, other: Tensor, **info) -> Tensor | None:
        # e.g. x + 1, 2.0 * x
        if isinstance(op, scalar_binops):
            return copy(other)  # keep the shape, dtype and device of the other
        elif isinstance(op, matmul_binop):
            self.add_diag_msg("matrix multiply is not supported", **info)
        return None

    def analyze_ewise_binop(self, op, left: Tensor, right: Tensor, **info) -> Tensor | None:
        # e.g. x + y, x * y
        if isinstance(op, scalar_binops):
            if left.shape != right.shape:
                self.add_diag_msg("element-wise binary operation requires the same shape", **info, left_shape=left.shape, right_shape=right.shape)
            if left.dtype != right.dtype:
                self.add_diag_msg("element-wise binary operation requires the same dtype", **info, left_dtype=left.dtype, right_dtype=right.dtype)
            if left.device != right.device:
                self.add_diag_msg("element-wise binary operation requires the same device", **info, left_device=left.device, right_device=right.device)
            return copy(left)  # keep the shape, dtype and device of the left
        # e.g. x @ y
        elif isinstance(op, matmul_binop):
            res_shape = Analyzer.calculate_matmul_shape(left.shape, right.shape)
            if res_shape is None:
                self.add_diag_msg("matrix multiply requires compatible shapes", **info, left_shape=left.shape, right_shape=right.shape)
            if left.dtype != right.dtype:
                self.add_diag_msg("matrix multiply requires the same dtype", **info, left_dtype=left.dtype, right_dtype=right.dtype)
            if left.device != right.device:
                self.add_diag_msg("matrix multiply requires the same device", **info, left_device=left.device, right_device=right.device)
            return Tensor(shape=res_shape, dtype=left.dtype, device=left.device)
        return None

    def analyze_binop(self, op, left, right, **info) -> Tensor | None:
        is_left_tensor, is_right_tensor = isinstance(left, Tensor), isinstance(right, Tensor)
        if is_left_tensor:
            if is_right_tensor:
                # both are tensors
                return self.analyze_ewise_binop(op, left, right, **info)
            else:
                # left is tensor, right is not tensor
                return self.analyze_scalar_binop(op, left, **info)
        elif is_right_tensor:
            # left is not tensor, right is tensor
            return self.analyze_scalar_binop(op, right, **info)
        # both are not tensors, we don't care
        return None

    def analyze_block_assert(self, retval: Tensor | None, output: Dict[str, Any], **info):
        if not isinstance(retval, Tensor):
            self.add_diag_msg("block_assert requires a tensor return value", **info)
            return
        if retval.shape !=output["shape"]:
            self.add_diag_msg("block_assert requires compatible shapes", **info, retval_shape=retval.shape, output_shape=output["shape"])
        if retval.dtype != output["dtype"]:
            self.add_diag_msg("block_assert requires the same dtype", **info, retval_dtype=retval.dtype, output_dtype=output["dtype"])
        if retval.device != output["device"]:
            self.add_diag_msg("block_assert requires the same device", **info, retval_device=retval.device, output_device=output["device"])