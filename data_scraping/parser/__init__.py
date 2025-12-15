"""
XML-Parser für Parlamentsdaten
Extrahiert strukturierte Daten aus XML-Dateien für ML-Modelle
"""

from .xml_parser import ParliamentXMLParser
from .models.parliament_models import (
    ParliamentSession,
    Speech,
    Speaker,
    Paragraph,
    Debate
)

__version__ = "1.0.0"
__author__ = "Parliament Data Processing System"

__all__ = [
    'ParliamentXMLParser',
    'ParliamentSession',
    'Speech', 
    'Speaker',
    'Paragraph',
    'Debate'
]
