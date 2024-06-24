# The `lbt.util` sub-package is largly copied from https://github.com/walkerning/aw_nas
import sys
import inspect

from lbt.utils.log import *
from lbt.utils.registry import *


def get_default_argspec(func):
    sig = inspect.signature(func)  # pylint: disable=no-member
    return [
        (n, param.default)
        for n, param in sig.parameters.items()
        if not param.default is param.empty
    ]


def _add_text_prefix(text, prefix):
    lines = text.split("\n")
    return "\n".join([prefix + line if line else line for line in lines])
