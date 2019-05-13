"""A Unix pipe-style command-line interface for audio processing.

This utility is heavily inspired by imagepipe:
https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py

The processing pipeline is a direct copy from the link above.
"""
from functools import update_wrapper


def processor(f):
    """Wrap a stream of processors."""
    def new_func(*args, **kwargs):
        def processor(stream):
            return f(stream, *args, **kwargs)
        return processor
    return update_wrapper(new_func, f)


def generator(f):
    """Similar to the processor.

    But passes through old values unchanged and does not pass through the
    values as parameter.
    """
    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in f(*args, **kwargs):
            yield item
    return update_wrapper(new_func, f)
