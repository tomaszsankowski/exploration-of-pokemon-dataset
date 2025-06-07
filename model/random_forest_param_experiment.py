import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import os
import time
import matplotlib.pyplot as plt
from itertools import product

RESULTS_DIR = 'model/random_forest_param_experiment/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Wczytaj dane
df_pokemon = pd.read_csv(
    'c:/Users/piotr/VsCodeProjects/exploration-of-pokemon-dataset/pokemon.csv')
df_combats = pd.read_csv(
    'c:/Users/piotr/VsCodeProjects/exploration-of-pokemon-dataset/combats.csv')
ID_COLUMN_NAME_IN_POKEMON_CSV = 'ID'

# Przygotowanie danych
pokemon_P1_stats = df_pokemon.add_suffix('_P1')
merged_df = pd.merge(df_combats, pokemon_P1_stats, left_on='First_pokemon',
                     right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P1')
pokemon_P2_stats = df_pokemon.add_suffix('_P2')
merged_df = pd.merge(merged_df, pokemon_P2_stats, left_on='Second_pokemon',
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
X = merged_df[features].fillna(0)
y = merged_df['Winner_Binary']

# Podział na train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Funkcja do swoistości


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float('nan')


# Parametry do testowania
n_estimators_list = [10, 50, 100, 200]
max_depth_list = [None, 5, 10, 20]
max_features_list = ['sqrt', 'log2', None]

results = []

for n_est, max_d, max_feat in product(n_estimators_list, max_depth_list, max_features_list):
    params = f"n_estimators={n_est}, max_depth={max_d}, max_features={max_feat}"
    start_train = time.time()
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                max_features=max_feat, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    train_time = time.time() - start_train
    start_pred = time.time()
    y_pred = rf.predict(X_val)
    predict_time = time.time() - start_pred
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred)
    spec = specificity_score(y_val, y_pred)
    results.append({
        'params': params,
        'n_estimators': n_est,
        'max_depth': max_d,
        'max_features': max_feat,
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'specificity': spec,
        'train_time': train_time,
        'predict_time': predict_time
    })
    with open(os.path.join(RESULTS_DIR, f'rf_{n_est}_{max_d}_{max_feat}_metrics.txt'), 'w') as f:
        f.write(f'{params}\nAccuracy: {acc:.4f}\nF1: {f1:.4f}\nRecall: {recall:.4f}\nSpecificity: {spec:.4f}\nTrain_time: {train_time:.4f}\nPredict_time: {predict_time:.4f}\n')

# Zapisz wyniki do CSV
with open(os.path.join(RESULTS_DIR, 'rf_param_experiment_results.csv'), 'w', newline='') as csvfile:
    fieldnames = ['params', 'n_estimators', 'max_depth', 'max_features',
                  'accuracy', 'f1', 'recall', 'specificity', 'train_time', 'predict_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Wykresy porównawcze (np. accuracy vs n_estimators dla różnych max_depth)
for max_feat in max_features_list:
    plt.figure(figsize=(10, 6))
    for max_d in max_depth_list:
        accs = [r['accuracy'] for r in results if r['max_features']
                == max_feat and r['max_depth'] == max_d]
        n_ests = [r['n_estimators'] for r in results if r['max_features']
                  == max_feat and r['max_depth'] == max_d]
        if accs:
            plt.plot(n_ests, accs, marker='o', label=f'max_depth={max_d}')
    plt.title(
        f'Random Forest Accuracy vs n_estimators (max_features={max_feat})')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(
        RESULTS_DIR, f'rf_accuracy_n_estimators_maxfeat_{max_feat}.png'))
    plt.close()

print('Eksperyment Random Forest zakończony. Wyniki i wykresy zapisane w', RESULTS_DIR)
