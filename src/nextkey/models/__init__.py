"""Model backbone registry and implementations."""

from nextkey.models.base import BaseCharTagger, create_model, list_models, register_model

# Import all backbones to trigger @register_model decorators
from nextkey.models import bigru  # noqa: F401
from nextkey.models import bilstm  # noqa: F401
from nextkey.models import cnn_tcn  # noqa: F401
from nextkey.models import cnn_bigru  # noqa: F401
from nextkey.models import tiny_transformer  # noqa: F401
from nextkey.models import tri_bigru  # noqa: F401
from nextkey.models import cascade_tri_bigru  # noqa: F401

__all__ = ["BaseCharTagger", "create_model", "list_models", "register_model"]
