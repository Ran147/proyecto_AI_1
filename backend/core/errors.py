"""Exceptions raised by the model registry and orchestrator.

Every exception carries the HTTP status code the API responds with, so the
routers never translate them one by one (see the handler in backend/main.py).
"""


class ModelError(Exception):
    """Base class for all model-related errors."""

    status_code = 500


class RegistrationError(ModelError):
    """A model is misconfigured; raised while the API starts."""


class DuplicateModelError(RegistrationError):
    """Two registered models share the same ``name``."""


class ModelNotFoundError(ModelError):
    """No model is registered under the requested name."""

    status_code = 404


class InvalidInputError(ModelError):
    """The inputs do not match the model's input schema."""

    status_code = 422


class ModelUnavailableError(ModelError):
    """The model exists but could not be loaded (e.g. missing artifact)."""

    status_code = 503
