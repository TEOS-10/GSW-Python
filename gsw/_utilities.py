from functools import wraps

import numpy as np

def masked_to_nan(arg):
    """
    Convert a masked array to a float ndarray with nans; ensure
    other arguments are float arrays or scalars.
    """
    if np.ma.isMaskedArray(arg):
        if arg.dtype.kind == 'f':
            return arg.filled(np.nan)
        else:
            return arg.astype(float).filled(np.nan)
    else:
        return np.asarray(arg, dtype=float)

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

        isarray = [hasattr(a, '__iter__') for a in args]
        ismasked = [np.ma.isMaskedArray(a) for a in args]
        isduck = [hasattr(a, '__array_ufunc__')
                    and not isinstance(a, np.ndarray) for a in args]

        hasarray = np.any(isarray)
        hasmasked = np.any(ismasked)
        hasduck = np.any(isduck)

        def fixup(ret):
            if hasduck:
                return ret
            if hasmasked:
                ret = np.ma.masked_invalid(ret)
            if not hasarray and isinstance(ret, np.ndarray) and ret.size == 1:
                try:
                    ret = ret[0]
                except IndexError:
                    pass
            return ret

        newargs = []
        for i, arg in enumerate(args):
            if ismasked[i]:
                newargs.append(masked_to_nan(arg))
            elif isduck[i]:
                newargs.append(arg)
            else:
                newargs.append(np.asarray(arg, dtype=float))

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


def axis_slicer(n, sl, axis):
    """
    Return an indexing tuple for an array with `n` dimensions,
    with slice `sl` taken on `axis`.
    """
    itup = [slice(None)] * n
    itup[axis] = sl
    return tuple(itup)


def indexer(shape, axis, order='C'):
    """
    Generator of indexing tuples for "apply_along_axis" usage.

    The generator cycles through all axes other than `axis`.
    The numpy np.apply_along_axis function only works with functions
    of a single array; this generator allows us work with a function
    of more than one array.
    """

    ndim = len(shape)
    ind_shape = list(shape)
    ind_shape[axis] = 1      # "axis" and any dim of 1 will not be incremented
    # list of indices, with a slice at "axis"
    inds = [0] * ndim
    inds[axis] = slice(None)
    kmax = np.prod(ind_shape)

    if order == 'C':
        index_position = list(reversed(range(ndim)))
    else:
        index_position = list(range(ndim))

    for k in range(kmax):
        yield tuple(inds)

        for i in index_position:
            if ind_shape[i] == 1:
                continue
            inds[i] += 1
            if inds[i] == ind_shape[i]:
                inds[i] = 0
            else:
                break


# This is straight from pycurrents.system.  We can trim out
# the parts we don't need, but there is no rush to do so.
class Bunch(dict):
    """
    A dictionary that also provides access via attributes.

    Additional methods update_values and update_None provide
    control over whether new keys are added to the dictionary
    when updating, and whether an attempt to add a new key is
    ignored or raises a KeyError.

    The Bunch also prints differently than a normal
    dictionary, using str() instead of repr() for its
    keys and values, and in key-sorted order.  The printing
    format can be customized by subclassing with a different
    str_ftm class attribute.  Do not assign directly to this
    class attribute, because that would substitute an instance
    attribute which would then become part of the Bunch, and
    would be reported as such by the keys() method.

    To output a string representation with
    a particular format, without subclassing, use the
    formatted() method.
    """

    str_fmt = "{0!s:<{klen}} : {1!s:>{vlen}}\n"

    def __init__(self, *args, **kwargs):
        """
        *args* can be dictionaries, bunches, or sequences of
        key,value tuples.  *kwargs* can be used to initialize
        or add key, value pairs.
        """
        dict.__init__(self)
        for arg in args:
            self.update(arg)
        self.update(kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError("'Bunch' object has no attribute '%s'" % name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        return self.formatted()

    def formatted(self, fmt=None, types=False):
        """
        Return a string with keys and/or values or types.

        *fmt* is a format string as used in the str.format() method.

        The str.format() method is called with key, value as positional
        arguments, and klen, vlen as kwargs.  The latter are the maxima
        of the string lengths for the keys and values, respectively,
        up to respective maxima of 20 and 40.
        """
        if fmt is None:
            fmt = self.str_fmt

        items = list(self.items())
        items.sort()

        klens = []
        vlens = []
        for i, (k, v) in enumerate(items):
            lenk = len(str(k))
            if types:
                v = type(v).__name__
            lenv = len(str(v))
            items[i] = (k, v)
            klens.append(lenk)
            vlens.append(lenv)

        klen = min(20, max(klens))
        vlen = min(40, max(vlens))
        slist = [fmt.format(k, v, klen=klen, vlen=vlen) for k, v in items]
        return ''.join(slist)

    def from_pyfile(self, filename):
        """
        Read in variables from a python code file.
        """
        # We can't simply exec the code directly, because in
        # Python 3 the scoping for list comprehensions would
        # lead to a NameError.  Wrapping the code in a function
        # fixes this.
        d = dict()
        lines = ["def _temp_func():\n"]
        with open(filename) as f:
            lines.extend(["    " + line for line in f])
        lines.extend(["\n    return(locals())\n",
                      "_temp_out = _temp_func()\n",
                      "del(_temp_func)\n"])
        codetext = "".join(lines)
        code = compile(codetext, filename, 'exec')
        exec(code, globals(), d)
        self.update(d["_temp_out"])
        return self

    def update_values(self, *args, **kw):
        """
        arguments are dictionary-like; if present, they act as
        additional sources of kwargs, with the actual kwargs
        taking precedence.

        One reserved optional kwarg is "strict".  If present and
        True, then any attempt to update with keys that are not
        already in the Bunch instance will raise a KeyError.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items() if k in self])
        self.update(dsub)

    def update_None(self, *args, **kw):
        """
        Similar to update_values, except that an existing value
        will be updated only if it is None.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items()
                                if k in self and self[k] is None])
        self.update(dsub)

    def _check_strict(self, strict, kw):
        if strict:
            bad = set(kw.keys()) - set(self.keys())
            if bad:
                bk = list(bad)
                bk.sort()
                ek = list(self.keys())
                ek.sort()
                raise KeyError(
                    "Update keys %s don't match existing keys %s" % (bk, ek))
