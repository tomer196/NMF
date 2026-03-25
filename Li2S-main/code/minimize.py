import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from scipy.optimize import minimize as _minimize
from typing import Dict, Tuple, Optional
from torch import cat, Tensor
from torch.autograd import grad

from inspect import signature, _empty


def _flatten(x: Tensor) -> Tensor:
    if isinstance(x, float):
        x = Tensor([x])
    elif isinstance(x, np.ndarray):
        x = Tensor(x)
    return x.reshape(-1)


class FuncWrapper:
    """
    A wrapper around tensor-valued functions allowing to flatten and unflatten arbitrarily-shaped
    arguments and calculate gradients.
    """
    
    def __init__(self, func, **kwargs):
        self._func = func
        
        # Store signature parameters
        params = signature(func).parameters
        self._params = tuple(params.keys())
        self._defaults = {
            k: v.default
            for k, v in params.items()
            if v.default != _empty
        }
        self._sizes: Optional[Dict[str, Tuple[int,...]]] = None
        self._fixed = kwargs
        self._reguired_params = set(p for p in self._params if p not in self._defaults)
        self._args = tuple(sorted(set(self._params) - set(self._fixed)))
        
    def set_sizes(self, sizes: Dict[str, Tuple[int,...]]):
        self._sizes = sizes
        
    def _getargs(self, validate: bool = True, *args, **kwargs) -> Dict[str, Tensor]:
    
        # List ordinal arguments and ensure they are not given again as kwargs
        ordinal = {name: val for name, val in zip(self._params, args)}
        
        if validate:
            dupe_args = set(kwargs.keys()) & (ordinal.keys())
            assert dupe_args == set(), f"Duplicate arguments {dupe_args}"
        
        params = {**ordinal, **kwargs, **self._fixed}
        
        # Ensure no arguments are missing 
        if validate:
            missing_args = set(self._args) - set(params.keys())
            assert missing_args == set(), \
                f"Missing required arguments {missing_args}"
        
        # Add default parameters
        all_params = {
            **params, 
            **{k: v for k, v in self._defaults.items() if k not in params}
        }
        
        return all_params
    
    def _argsizes(self, args: Dict[str, Tensor]) -> Dict[str, Tuple[int,...]]:
        return {k: tuple(v.shape) for k, v in args.items()}
    
    def argsizes(self, *args, **kwargs) -> Dict[str, Tuple[int,...]]:
        return self._argsizes(
            {
                k: v
                for k, v in self._getargs(True, *args, **kwargs).items()
                if k not in self._fixed
            }
        )

    def flatten(self, *args, **kwargs) -> Tensor:
        return self._flatten(
            self._getargs(True, *args, **kwargs)
        )
        
    def _flatten(self, params: Dict[str, Tensor]) -> Tensor:
        # Flat-iron everything and concatenate into a spaghetti vector
        return cat([_flatten(params[k]) for k in self._args])
        
    def unflatten(self, x: Tensor, sizes: Optional[Dict[str, Tuple[int,...]]]=None):
        sizes = sizes or self._sizes
        assert sizes
        xlen = [
            np.prod(sizes[k]) for k in self._args
        ]
        return {
            key: x[i:i+n].reshape(shape)
            for i, n, (key, shape) in zip(np.cumsum([0, *xlen]), xlen, sizes.items())
        }
    
    def func(self, x: Tensor, sizes: Optional[Dict[str, Tuple[int,...]]]=None) -> Tensor:
        sizes = sizes or self._sizes
        assert sizes
        x = x if isinstance(x, Tensor) else Tensor(x)
        return self._func(**self.unflatten(x, sizes), **self._fixed)
    
    def __call__(self, x: Tensor, sizes: Optional[Dict[str, Tuple[int,...]]]=None) -> Tensor:
        f = self.func(x, sizes)
        return f.numpy()
        
    def grad(self, x: Tensor, sizes: Optional[Dict[str, Tuple[int,...]]]=None) -> Tensor:
        sizes = sizes or self._sizes
        assert sizes
        x = x if isinstance(x, Tensor) else Tensor(x)
        x = x.clone().detach()
        x.requires_grad_()
        f = self.func(x, sizes)
        g = grad(f, x)
        return g[0].numpy()
        
        
def minimize(func, X0: Dict[str, Tensor], args: Dict[str, Tensor]={}, *more_args, **kwargs):
    """
    Wrapper around scipy.optimize.minimize.
    Allows to receive multiple arguments as a dictionary of named tensors. 
    
    Example:
    def f(X: Tensor, Y: Tensor, z: Tensor) -> Tensor:
        ...
        
    result = minimize(f, {'X': randn(5,3), 'z': randn(3)}, args={'Y': eye(5)}, ...)
    """
    
    f = FuncWrapper(func, **args)
    x0 = f.flatten(**X0).numpy()
    sz = f.argsizes(**X0)
    f.set_sizes(sz)
    result = _minimize(f, x0, (), *more_args, **kwargs, jac=f.grad)
    result.x = {
        k: Tensor(v)
        for k, v in f.unflatten(result.x, sz).items()
    }
    return result

