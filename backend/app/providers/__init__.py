"""Live data provider abstraction layer.

DemoProvider has been retired from production (Wave 9b) and lives at
`app/tests/fixtures/demo_provider.py` as a test-only fixture. All
runtime callers should go through `services.data_service` rather than
importing providers directly.
"""
from .base import BaseProvider, ProviderStatus

__all__ = ["BaseProvider", "ProviderStatus"]
