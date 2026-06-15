"""
Démonstration d'inférence
À exécuter: python notebooks/03_inference_demo.py
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json

from src.models import TransformerRCA
from src.inference import RCAInference

print("=" * 80)
print("DÉMONSTRATION D'INFÉRENCE")
print("=" * 80)

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/best_model.pth'
METADATA_PATH = 'models/metadata.json'

# 1. Chargement des métadonnées
print("\n1️⃣ CHARGEMENT DES MÉTADONNÉES...")
if Path(METADATA_PATH).exists():
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    class_names = metadata['class_names']
    print(f"✓ Métadonnées chargées")
    print(f"  Classes: {class_names}")
else:
    class_names = ['network_failure', 'memory_leak', 'disk_full', 
                   'authentication_error', 'resource_exhaustion']
    print(f"⚠️  Métadonnées non trouvées, utilisation des classes par défaut")

# 2. Chargement du modèle
print("\n2️⃣ CHARGEMENT DU MODÈLE...")
model = TransformerRCA(num_classes=len(class_names))
model.to(DEVICE)

if Path(MODEL_PATH).exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✓ Modèle chargé: {MODEL_PATH}")
else:
    print(f"⚠️  Modèle non trouvé: {MODEL_PATH}")
    print(f"   Utilisation d'un modèle non entraîné")

# 3. Initialisation de l'inferencer
print("\n3️⃣ INITIALISATION DE L'INFERENCER...")
inferencer = RCAInference(model, device=DEVICE, class_names=class_names)
print(f"✓ Inferencer prêt (device: {DEVICE})")

# 4. Prédictions de test
print("\n4️⃣ PRÉDICTIONS DE TEST...")

test_logs = [
    "ERROR Connection timeout to 192.168.1.100:8080",
    "WARNING Memory usage at 95% on node-001",
    "CRITICAL Disk space low on /var: 99% used",
    "ERROR Authentication failed for user admin",
    "ERROR Thread pool exhausted, 500 tasks pending"
]

for i, log in enumerate(test_logs, 1):
    print(f"\n[Log {i}]")
    print(f"Text: {log}")
    
    result = inferencer.predict_single(log, threshold=0.5)
    
    print(f"Prédiction: {result['predicted_class']}")
    print(f"Confiance: {result['confidence']:.2%}")
    
    # Top-k
    top_k = inferencer.get_top_k_predictions(log, k=3)
    print(f"Top-3:")
    for j, (cls, prob) in enumerate(top_k, 1):
        print(f"  {j}. {cls}: {prob:.2%}")

# 5. Explication détaillée
print("\n5️⃣ EXPLICATION DÉTAILLÉE...")
test_log = test_logs[0]
explanation = inferencer.explain_prediction(test_log)

print(f"\nLog: {test_log}")
print(f"Prédiction: {explanation['predicted_class']}")
print(f"Confiance: {explanation['confidence']:.2%}")
print(f"Distribution de probabilité:")
for cls, prob in explanation['probabilities'].items():
    bar = "█" * int(prob * 50)
    print(f"  {cls:25} {prob:5.1%} {bar}")

print("\n" + "=" * 80)
print("✅ DÉMONSTRATION TERMINÉE")
print("=" * 80)
