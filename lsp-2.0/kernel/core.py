from __future__ import annotations
from typing import Dict, Any, Tuple


class Variable:
    # Base class for Tensor & BasicBlock
    pass

class BasicBlock(Variable):
    # represents an instance of BasicBlock
    # decorated by our custom decorator
    # e.g Conv2d, ReLU, MaxPool2d, etc
    _name: str
    _kwargs: Dict[str, Any]

    def __init__(self, name, kwargs=None):
        super().__init__()
        self._name = name
        self._kwargs = kwargs if kwargs is not None else {}

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._kwargs

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"<BasicBlock: {self.name}, {self.kwargs}>"

    def copy(self) -> BasicBlock:
        return BasicBlock(name=self.name, kwargs=self.kwargs)


class Tensor(Variable):
    # represents an instance of Tensor
    _shape: Tuple[int]
    _dtype: str
    _device: str

    def __eq__(self, other: Tensor) -> bool:
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def __init__(self, shape=None, dtype=None, device=None):
        super().__init__()
        self._shape = shape
        self._dtype = dtype
        self._device = device

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    def __repr__(self) -> str:
        return f"<Tensor, shape={self.shape}, dtype={self.dtype}, device={self.device}>"

    def copy(self) -> Tensor:
        return Tensor(shape=self.shape, dtype=self.dtype, device=self.device)
