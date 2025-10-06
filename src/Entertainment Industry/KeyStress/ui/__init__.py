"""
User interface module for KeyStress.

This module provides command-line and web-based interfaces for
interacting with the KeyStress system.
"""

from ui.cli import CLI
from ui.dashboard import Dashboard

__all__ = ['CLI', 'Dashboard']
