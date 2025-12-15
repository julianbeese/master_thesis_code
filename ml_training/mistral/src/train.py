"""
Training Script für GPT-OSS-20B Fine-Tuning mit LoRA
Multi-Label-Klassifikation für Brexit Debate Daten
"""
import os
import yaml
import torch
# import wandb  # Disabled for now
from pathlib import Path
from typing import Dict, Optional
from huggingface_hub import login
from dotenv import load_dotenv
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
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

from data_loader import CSVDataLoader


# Nutze Standard Trainer - CrossEntropyLoss wird automatisch für Multi-Class verwendet


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


def setup_model_and_tokenizer(config: Dict, label_info: Dict, hf_token: str = None):
    """Initialisiert Model und Tokenizer mit LoRA"""
    model_name = config['model']['base_model']
    num_labels = label_info['num_labels']
    label2id = label_info['label2id']
    id2label = label_info['id2label']
    
    print(f"Lade Model: {model_name}")
    
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model']['trust_remote_code'],
        token=hf_token
    )
    
    # Padding Token setzen falls nicht vorhanden
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Model für Multi-Class-Klassifikation laden (speichereffizient)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
        trust_remote_code=config['model']['trust_remote_code'],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,  # Disabled for now
        load_in_4bit=torch.cuda.is_available(),  # Only use 4-bit on GPU
        token=hf_token
    )
    
    # Explicitly move model to GPU if available
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Model moved to GPU: {device}")
    else:
        print("WARNING: CUDA not available, running on CPU!")
    
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
    # GPU-optimized settings
    use_gpu = torch.cuda.is_available()
    
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
        fp16=config['training']['fp16'] and use_gpu,  # Only use FP16 on GPU
        bf16=config['training']['bf16'] and use_gpu,  # Only use BF16 on GPU
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
        dataloader_pin_memory=use_gpu,  # Pin memory only on GPU
        dataloader_num_workers=4 if use_gpu else 0,  # More workers on GPU
    )


def main():
    # Lade Umgebungsvariablen aus .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Umgebungsvariablen aus .env geladen")
    else:
        print("⚠️  .env nicht gefunden, verwende System-Umgebungsvariablen")
    
    # Hugging Face Token für gated models
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN nicht gefunden!")
        print("   Erstelle .env mit HF_TOKEN=dein_token")
        print("   oder setze HF_TOKEN='dein_token' als Umgebungsvariable")
        hf_token = None
    else:
        login(token=hf_token)
        print("✅ Hugging Face Token authentifiziert")
    
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
    #     project="gpt-oss-20b-brexit-debates",
    #     config=config,
    #     name=f"gpt-oss-20b-lora-{config['training']['learning_rate']}"
    # )
    
    # Daten laden
    print("Lade Daten...")
    data_loader = CSVDataLoader(str(config_path))
    label_info = data_loader.get_label_info()
    num_labels = label_info['num_labels']
    
    print(f"\nAnzahl Labels: {num_labels}")
    print(f"Label Namen: {label_info['label_names']}")
    
    # Model und Tokenizer initialisieren
    model, tokenizer = setup_model_and_tokenizer(config, label_info, hf_token)
    
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
    print("Starte Training...")
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
    print("Training abgeschlossen!")
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
