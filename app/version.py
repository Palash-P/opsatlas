"""Single authoritative application version.

``pyproject.toml`` reads this module dynamically, so the installed distribution
metadata, the FastAPI/OpenAPI metadata, evaluation reports, and any Docker image
built from this project all agree. Do not hardcode a version elsewhere.

Changing this value is a release action. It does not change the database schema
or the API contract.
"""

from __future__ import annotations

APP_VERSION = "0.6.0"
__version__ = APP_VERSION
