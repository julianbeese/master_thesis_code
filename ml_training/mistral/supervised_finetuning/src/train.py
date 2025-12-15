"""
Training Script für Supervised Fine-Tuning mit LoRA
Frame-Classification für Brexit Debate Daten
"""
import os
import yaml
import torch
# import wandb  # Disabled for now
from pathlib import Path
from typing import Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

from data_loader import SFTDataLoader


def compute_metrics(eval_pred):
    """Metriken für Multi-Class-Klassifikation"""
    predictions, labels = eval_pred
    
    # Argmax für predicted class
    predictions = np.argmax(predictions, axis=1)
    
    # Metriken berechnen
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def setup_model_and_tokenizer(config: Dict, label_info: Dict):
    """Initialisiert Model und Tokenizer mit LoRA"""
    model_name = config['model']['base_model']
    num_labels = label_info['num_labels']
    label2id = label_info['label2id']
    id2label = label_info['id2label']
    
    print(f"Lade Model: {model_name}")
    
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    # Padding Token setzen falls nicht vorhanden
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model für Multi-Class-Klassifikation laden (speichereffizient)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
        trust_remote_code=config['model']['trust_remote_code'],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        load_in_4bit=True,  # 4-bit quantization for maximum memory efficiency
        low_cpu_mem_usage=True
    )
    
    # Move model to device
    model = model.to(device)
    
    # Set pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # LoRA Configuration
    if config['training']['use_lora']:
        print("Konfiguriere LoRA...")
        
        # Prepare model for k-bit training if using quantization
        if hasattr(model, 'is_quantized') and model.is_quantized:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=config['training']['lora_r'],
            lora_alpha=config['training']['lora_alpha'],
            target_modules=config['training']['lora_target_modules'],
            lora_dropout=config['training']['lora_dropout'],
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def setup_training_args(config: Dict, output_dir: Path) -> TrainingArguments:
    """Erstellt TrainingArguments aus Config"""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['evaluation']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        optim=config['training']['optimizer'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=config['evaluation']['metric'],
        greater_is_better=True,
        report_to=None,  # Disabled W&B for now
        logging_first_step=True,
        remove_unused_columns=False,
    )


def main():
    # Config laden
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ausgabe-Verzeichnisse erstellen
    output_dir = Path(__file__).parent.parent / config['training']['output_dir']
    checkpoint_dir = Path(__file__).parent.parent / config['training']['checkpoint_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Weights & Biases initialisieren (disabled for now)
    # wandb.init(
    #     project="supervised-finetuning-brexit-debates",
    #     config=config,
    #     name=f"mistral-7b-sft-{config['training']['learning_rate']}"
    # )
    
    # Daten laden
    print("Lade Daten...")
    data_loader = SFTDataLoader(str(config_path))
    label_info = data_loader.get_label_info()
    num_labels = label_info['num_labels']
    
    print(f"\nAnzahl Labels: {num_labels}")
    print(f"Label Namen: {label_info['label_names']}")
    
    # Model und Tokenizer initialisieren
    model, tokenizer = setup_model_and_tokenizer(config, label_info)
    
    # Datasets erstellen
    print("Erstelle Datasets...")
    train_dataset, validation_dataset, test_dataset = data_loader.prepare_datasets(tokenizer)
    
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(validation_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")
    
    # Training Arguments
    training_args = setup_training_args(config, output_dir)
    
    # Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Training starten
    print("\n" + "="*50)
    print("Starte Supervised Fine-Tuning...")
    print("="*50 + "\n")
    
    train_result = trainer.train()
    
    # Metriken speichern
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Model speichern
    print("\nSpeichere finales Model...")
    trainer.save_model(str(checkpoint_dir / "final_model"))
    tokenizer.save_pretrained(str(checkpoint_dir / "final_model"))
    
    # Evaluation auf Validation Set
    print("\nEvaluiere Model auf Validation Set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Finale Evaluation auf Test Set
    print("\nFinale Evaluation auf Test Set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    print("\n" + "="*50)
    print("Supervised Fine-Tuning abgeschlossen!")
    print("="*50)
    print(f"\nValidation Metriken:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nTest Metriken:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # wandb.finish()  # Disabled for now


if __name__ == "__main__":
    main()
