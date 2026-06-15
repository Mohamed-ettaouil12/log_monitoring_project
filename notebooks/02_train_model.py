"""
Script d'entraînement du modèle AMÉLIORÉ
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

from src.models import TransformerRCA
from src.training import Trainer
from src.preprocessing import LogPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("ENTRAÎNEMENT DU MODÈLE TRANSFORMER (VERSION AMÉLIORÉE)")
print("=" * 80)

# Configuration AMÉLIORÉE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8  # Plus petit = meilleur apprentissage
EPOCHS = 15  # Plus d'epochs pour meilleure convergence
LEARNING_RATE = 1e-5  # Plus petit = plus stable
WARMUP_STEPS = 1000  # Plus de warmup

print(f"\n📱 Environnement:")
print(f"  Device: {DEVICE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

# 1. Charger les données
print("\n1️⃣ CHARGEMENT DES DONNÉES...")
df = pd.read_csv('data/raw/synthetic_logs.csv')
print(f"✓ {len(df)} logs chargés")
print(f"  Distribution:")
print(df['label'].value_counts())

# 2. Encoder les labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
class_names = le.classes_.tolist()

print(f"\n✓ {len(class_names)} classes:")
for i, name in enumerate(class_names):
    count = len(df[df['label'] == name])
    print(f"    {i}: {name} ({count} samples)")

# 3. Préparation
print("\n2️⃣ PRÉPARATION DES DONNÉES...")
preprocessor = LogPreprocessor(max_length=256)
preprocessor.build_vocab(df['log'].tolist(), min_freq=1)

X_encoded = np.array([preprocessor.encode(log) for log in df['log']])
y_encoded = df['label_encoded'].values

# 4. Split
indices = np.arange(len(df))
train_idx, temp_idx = train_test_split(indices, test_size=0.25, random_state=42, stratify=y_encoded)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_encoded[temp_idx])

X_train, X_val, X_test = X_encoded[train_idx], X_encoded[val_idx], X_encoded[test_idx]
y_train, y_val, y_test = y_encoded[train_idx], y_encoded[val_idx], y_encoded[test_idx]

print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 5. DataLoaders
print("\n3️⃣ CRÉATION DES DATALOADERS...")
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.long),
    torch.zeros_like(torch.tensor(X_train, dtype=torch.long)),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.long),
    torch.zeros_like(torch.tensor(X_val, dtype=torch.long)),
    torch.tensor(y_val, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✓ Train loader: {len(train_loader)} batches")
print(f"✓ Val loader: {len(val_loader)} batches")

# 6. Modèle
print("\n4️⃣ INITIALISATION DU MODÈLE...")
model = TransformerRCA(
    model_name="distilbert-base-uncased",
    num_classes=len(class_names),
    dropout=0.3
)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modèle créé: {total_params:,} paramètres")

# 7. Entraînement
print("\n5️⃣ ENTRAÎNEMENT...")
trainer = Trainer(model, device=DEVICE, output_dir='models')

history = trainer.train(
    train_loader,
    val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS
)

# 8. Sauvegarder
print("\n6️⃣ SAUVEGARDE DES RÉSULTATS...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
axes[0].set_title('Loss par époque', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['val_accuracy'], label='Accuracy', marker='o')
axes[1].plot(history['val_f1'], label='F1-Score', marker='o')
axes[1].set_title('Métriques par époque', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/03_training_history.png', dpi=150, bbox_inches='tight')
print("✓ Graphique: results/03_training_history.png")

# 9. Métadonnées
print("\n7️⃣ SAUVEGARDE DES MÉTADONNÉES...")
import json

metadata = {
    'model_type': 'TransformerRCA',
    'model_name': 'distilbert-base-uncased',
    'num_classes': len(class_names),
    'class_names': class_names,
    'vocab_size': preprocessor.vocab_size,
    'max_length': preprocessor.max_length,
    'epochs_trained': EPOCHS,
    'final_metrics': {
        'train_loss': float(history['train_loss'][-1]),
        'val_loss': float(history['val_loss'][-1]),
        'val_accuracy': float(history['val_accuracy'][-1]),
        'val_f1': float(history['val_f1'][-1])
    }
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Métadonnées: models/metadata.json")

print("\n" + "=" * 80)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("=" * 80)
