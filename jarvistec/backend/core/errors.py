"""Errores del orquestador de modelos.

Cada error trae el `status_code` HTTP con el que la API lo reporta; los routers
no necesitan traducirlos uno por uno (ver el manejador en backend/main.py).
"""


class ModelError(Exception):
    status_code = 500


class RegistrationError(ModelError):
    """Configuración inválida de un modelo al registrarlo (falla al iniciar la API)."""


class DuplicateModelError(RegistrationError):
    """Dos modelos registrados con el mismo `name`."""


class ModelNotFoundError(ModelError):
    status_code = 404


class InvalidInputError(ModelError):
    status_code = 422


class ModelUnavailableError(ModelError):
    """El modelo existe pero no se pudo cargar (p. ej. falta su artefacto en artifacts/)."""
    status_code = 503
