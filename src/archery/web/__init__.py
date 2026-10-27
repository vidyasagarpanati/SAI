"""The local web UI. A front end over the CLI, not a second implementation."""
from archery.web.app import create_app

__all__ = ["create_app"]
