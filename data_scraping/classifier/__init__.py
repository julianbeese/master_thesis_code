"""
Brexit Classifier Package
Intelligente Klassifikation von Parliament-Debatten auf Brexit-Bezug mit Google Gemini AI
"""

from .brexit_classifier import BrexitClassifier, create_brexit_classifier, BrexitClassificationResult
from .brexit_database import BrexitDatabase, create_brexit_database
from .brexit_pipeline import BrexitPipeline, create_brexit_pipeline

__version__ = "1.0.0"
__author__ = "Parliament Data Analysis Team"

__all__ = [
    "BrexitClassifier",
    "create_brexit_classifier", 
    "BrexitClassificationResult",
    "BrexitDatabase",
    "create_brexit_database",
    "BrexitPipeline",
    "create_brexit_pipeline"
]
