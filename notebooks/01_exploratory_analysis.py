"""
Analyse exploratoire des données (EDA)
À exécuter: python notebooks/01_exploratory_analysis.py
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import LogDataGenerator
from src.preprocessing import LogPreprocessor

# Configuration
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("ANALYSE EXPLORATOIRE DES DONNÉES")
print("=" * 80)

# 1. Génération du dataset
print("\n1️⃣ GÉNÉRATION DU DATASET...")
generator = LogDataGenerator()
df = generator.generate_dataset(n_samples=1000)

print(f"✓ Dataset généré: {len(df)} logs")
print(f"✓ Colonnes: {df.columns.tolist()}")
print(f"\nPremiers logs:")
print(df.head())

# 2. Analyse des labels
print("\n2️⃣ DISTRIBUTION DES CAUSES RACINES...")
label_dist = df['label'].value_counts()
print(label_dist)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme
label_dist.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Distribution des causes racines', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Nombre de logs')
axes[0].set_xlabel('Cause racine')
axes[0].tick_params(axis='x', rotation=45)

# Pie chart
axes[1].pie(label_dist, labels=label_dist.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Proportion des causes racines', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/01_label_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Graphique sauvegardé: results/01_label_distribution.png")

# 3. Statistiques des logs
print("\n3️⃣ STATISTIQUES DES LOGS...")
log_lengths = df['log'].str.len()
token_counts = df['log'].str.split().str.len()

print(f"Longueur moyenne: {log_lengths.mean():.0f} caractères")
print(f"Longueur max: {log_lengths.max():.0f} caractères")
print(f"Tokens moyen: {token_counts.mean():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(log_lengths, bins=30, color='lightcoral', edgecolor='black')
axes[0].set_title('Distribution des longueurs de logs', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Nombre de caractères')
axes[0].set_ylabel('Fréquence')

axes[1].hist(token_counts, bins=30, color='lightgreen', edgecolor='black')
axes[1].set_title('Distribution du nombre de tokens', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Nombre de tokens')
axes[1].set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig('results/02_log_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Graphique sauvegardé: results/02_log_statistics.png")

# 4. Analyse du prétraitement
print("\n4️⃣ IMPACT DU PRÉTRAITEMENT...")
preprocessor = LogPreprocessor()

df['log_cleaned'] = df['log'].apply(preprocessor.clean_log)
df['length_original'] = df['log'].str.len()
df['length_cleaned'] = df['log_cleaned'].str.len()

print(f"Longueur moyenne avant: {df['length_original'].mean():.0f}")
print(f"Longueur moyenne après: {df['length_cleaned'].mean():.0f}")
print(f"Réduction: {(1 - df['length_cleaned'].mean()/df['length_original'].mean()) * 100:.1f}%")

print("\nExemples de nettoyage:")
for i in range(3):
    print(f"\nOriginal:  {df['log'].iloc[i]}")
    print(f"Cleaned:   {df['log_cleaned'].iloc[i]}")

# 5. Sauvegarde du dataset
print("\n5️⃣ SAUVEGARDE DU DATASET...")
df.to_csv('data/raw/synthetic_logs.csv', index=False)
print("✓ Dataset sauvegardé: data/raw/synthetic_logs.csv")

# 6. Résumé
print("\n" + "=" * 80)
print("RÉSUMÉ DE L'EDA")
print("=" * 80)
print(f"✓ {len(df)} logs générés")
print(f"✓ {len(df['label'].unique())} causes racines")
print(f"✓ Longueur moyenne: {log_lengths.mean():.0f} caractères")
print(f"✓ Distribution équilibrée: {label_dist.std() < 50}")
print(f"✓ Dataset sauvegardé dans data/raw/")
print("=" * 80)
