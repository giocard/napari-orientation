try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .orientation_field_widget import vector_field_widget

__all__ = ("vector_field_widget",)
