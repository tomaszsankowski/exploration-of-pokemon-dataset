from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump, load
import numpy as np
import os

# Zmienna na folder do zapisu wyników
RESULTS_DIR = 'model/supervised/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Wczytaj dane i przygotuj je tak samo jak w modelu głównym
pokemon_df = pd.read_csv(
    'c:/Users/piotr/VsCodeProjects/exploration-of-pokemon-dataset/pokemon.csv')
combats_df = pd.read_csv(
    'c:/Users/piotr/VsCodeProjects/exploration-of-pokemon-dataset/combats.csv')

ID_COLUMN_NAME_IN_POKEMON_CSV = 'ID'

pokemon_P1_stats = pokemon_df.add_suffix('_P1')
merged_df = pd.merge(combats_df, pokemon_P1_stats,
                     left_on='First_pokemon',
                     right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P1')
pokemon_P2_stats = pokemon_df.add_suffix('_P2')
merged_df = pd.merge(merged_df, pokemon_P2_stats,
                     left_on='Second_pokemon',
                     right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P2')

stats_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
for stat in stats_columns:
    merged_df[f'Diff_{stat}'] = merged_df[f'{stat}_P1'] - \
        merged_df[f'{stat}_P2']
merged_df['Legendary_P1'] = merged_df['Legendary_P1'].astype(int)
merged_df['Legendary_P2'] = merged_df['Legendary_P2'].astype(int)
merged_df['Winner_Binary'] = merged_df.apply(
    lambda row: 0 if row['Winner'] == row['First_pokemon'] else 1, axis=1)
features = [f'Diff_{stat}' for stat in stats_columns] + \
    ['Legendary_P1', 'Legendary_P2']
X = merged_df[features]
y = merged_df['Winner_Binary']
X = X.fillna(X.mean())

# Podział na train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Załaduj model
model = load(RESULTS_DIR + 'pokemon_rf_model.joblib')

# --- Analiza modelu ---
# 1. Ważność cech
importances = model.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig(RESULTS_DIR + 'feature_importances.png')
plt.close()

# 2. Przykładowe drzewo decyzyjne
plt.figure(figsize=(20, 8))
plot_tree(model.estimators_[0], feature_names=features,
          filled=True, max_depth=3, fontsize=8)
plt.title('Sample Decision Tree (depth=3)')
plt.savefig(RESULTS_DIR + 'sample_tree.png')
plt.close()

# 3. Macierz pomyłek i raport klasyfikacji
val_pred = model.predict(X_val)
cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(RESULTS_DIR + 'confusion_matrix.png')
plt.close()

with open(RESULTS_DIR + 'classification_report.txt', 'w') as f:
    f.write(classification_report(y_val, val_pred))

print('Analiza modelu nadzorowanego zakończona. Wyniki zapisane w folderze', RESULTS_DIR)
