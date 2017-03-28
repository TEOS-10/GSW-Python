from functools import wraps

import numpy as np

def masked_to_nan(arg):
    """
    Convert a masked array to a float ndarray with nans; leave other
    objects unchanged.
    """
    if np.ma.isMaskedArray(arg):
        if arg.dtype.kind == 'f':
            return arg.filled(np.nan)
        else:
            return arg.astype(float).filled(np.nan)
    else:
        return arg

def match_args_return(f):
    """
    Decorator for most functions that operate on profile data.
    """
    @wraps(f)
    def wrapper(*args, **kw):
        p = kw.get('p', None)
        if p is not None:
            args = list(args)
            args.append(p)

        isarray = np.any([hasattr(a, '__iter__') for a in args])
        ismasked = np.any([np.ma.isMaskedArray(a) for a in args])

        def fixup(ret):
            if ismasked:
                ret = np.ma.masked_invalid(ret)
            if not isarray:
                ret = ret[0]
            return ret

        if ismasked:
            newargs = [masked_to_nan(a) for a in args]
        else:
            newargs = args

        if p is not None:
            kw['p'] = newargs.pop()

        ret = f(*newargs, **kw)

        if isinstance(ret, tuple):
            retlist = [fixup(arg) for arg in ret]
            ret = tuple(retlist)
        else:
            ret = fixup(ret)
        return ret
    wrapper.__wrapped__ = f
    return wrapper
