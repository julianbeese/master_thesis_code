"""
Data Loader für Supervised Fine-Tuning
Lädt CSV-Daten für Frame-Classification Training
"""
import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer


class SFTDataLoader:
    def __init__(self, config_path: str):
        """
        Initialisiert den Data Loader
        
        Args:
            config_path: Pfad zur YAML-Konfigurationsdatei
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.base_path = Path(config_path).resolve().parent.parent
        
    def get_label_info(self) -> Dict:
        """
        Gibt Label-Informationen zurück
        
        Returns:
            Dictionary mit Label-Mapping und Anzahl
        """
        label2id = self.data_config['label2id']
        id2label = {v: k for k, v in label2id.items()}
        
        return {
            'num_labels': self.data_config['num_labels'],
            'label2id': label2id,
            'id2label': id2label,
            'label_names': list(label2id.keys())
        }
    
    def load_csv_data(self, csv_file: str) -> pd.DataFrame:
        """
        Lädt CSV-Datei
        
        Args:
            csv_file: Name der CSV-Datei
            
        Returns:
            DataFrame mit geladenen Daten
        """
        data_dir = Path(self.data_config['data_dir'])
        if not data_dir.is_absolute():
            data_dir = self.base_path / data_dir
        
        csv_path = data_dir / csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")
        
        print(f"Lade CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Prüfe ob erforderliche Spalten vorhanden sind
        text_col = self.data_config['text_column']
        label_col = self.data_config['label_column']
        
        if text_col not in df.columns:
            raise ValueError(f"Text-Spalte '{text_col}' nicht in CSV gefunden")
        if label_col not in df.columns:
            raise ValueError(f"Label-Spalte '{label_col}' nicht in CSV gefunden")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Dataset:
        """
        Bereitet DataFrame für Training vor
        
        Args:
            df: DataFrame mit Text und Labels
            
        Returns:
            Hugging Face Dataset
        """
        text_col = self.data_config['text_column']
        label_col = self.data_config['label_column']
        label2id = self.data_config['label2id']
        
        # Filtere fehlende Werte
        df = df.dropna(subset=[text_col, label_col])
        
        # Konvertiere Labels zu IDs
        df['label'] = df[label_col].map(label2id)
        
        # Entferne Zeilen mit unbekannten Labels
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        # Erstelle Dataset
        dataset_dict = {
            'text': df[text_col].tolist(),
            'label': df['label'].tolist()
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        return dataset
    
    def tokenize_function(self, examples: Dict, tokenizer: PreTrainedTokenizer) -> Dict:
        """
        Tokenisiert Texte
        
        Args:
            examples: Dictionary mit Texten
            tokenizer: Hugging Face Tokenizer
            
        Returns:
            Tokenisierte Texte
        """
        max_length = self.data_config['max_length']
        
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
    
    def prepare_datasets(self, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Lädt und bereitet Train-, Validation- und Test-Datasets vor
        
        Args:
            tokenizer: Hugging Face Tokenizer
            
        Returns:
            Tuple von (train_dataset, validation_dataset, test_dataset)
        """
        # Lade CSVs
        train_df = self.load_csv_data(self.data_config['train_csv'])
        validation_df = self.load_csv_data(self.data_config['validation_csv'])
        test_df = self.load_csv_data(self.data_config['test_csv'])
        
        print(f"Train Zeilen: {len(train_df)}")
        print(f"Validation Zeilen: {len(validation_df)}")
        print(f"Test Zeilen: {len(test_df)}")
        
        # Bereite Datasets vor
        train_dataset = self.prepare_data(train_df)
        validation_dataset = self.prepare_data(validation_df)
        test_dataset = self.prepare_data(test_df)
        
        print(f"Train Samples nach Filterung: {len(train_dataset)}")
        print(f"Validation Samples nach Filterung: {len(validation_dataset)}")
        print(f"Test Samples nach Filterung: {len(test_dataset)}")
        
        # Tokenisiere
        print("Tokenisiere Train-Daten...")
        train_dataset = train_dataset.map(
            lambda examples: self.tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=['text']
        )
        
        print("Tokenisiere Validation-Daten...")
        validation_dataset = validation_dataset.map(
            lambda examples: self.tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=['text']
        )
        
        print("Tokenisiere Test-Daten...")
        test_dataset = test_dataset.map(
            lambda examples: self.tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=['text']
        )
        
        # Setze Format für PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return train_dataset, validation_dataset, test_dataset
