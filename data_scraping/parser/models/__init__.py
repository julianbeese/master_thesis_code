"""
Datenmodelle für Parlamentsdaten
"""

from .parliament_models import (
    ParliamentSession,
    Speech,
    Speaker,
    Paragraph,
    Debate
)

__all__ = [
    'ParliamentSession',
    'Speech',
    'Speaker', 
    'Paragraph',
    'Debate'
]
