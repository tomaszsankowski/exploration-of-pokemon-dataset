import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import os

RESULTS_DIR = 'model/decision_tree_depth_experiment/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Wczytaj dane
pokemon_df = pd.read_csv(
    'c:/Users/piotr/VsCodeProjects/exploration-of-pokemon-dataset/pokemon.csv')
combats_df = pd.read_csv(
    'c:/Users/piotr/VsCodeProjects/exploration-of-pokemon-dataset/combats.csv')
ID_COLUMN_NAME_IN_POKEMON_CSV = 'ID'

# Przygotowanie danych
pokemon_P1_stats = pokemon_df.add_suffix('_P1')
merged_df = pd.merge(combats_df, pokemon_P1_stats, left_on='First_pokemon',
                     right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P1')
pokemon_P2_stats = pokemon_df.add_suffix('_P2')
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
X = merged_df[features]
y = merged_df['Winner_Binary']
X = X.fillna(X.mean())

# Podział na train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Eksperyment: różne głębokości drzewa
max_depths = list(range(1, 21))
accuracies = []
f1s = []
recalls = []
precisions = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    accuracies.append(acc)
    f1s.append(f1)
    recalls.append(recall)
    precisions.append(precision)
    with open(os.path.join(RESULTS_DIR, f'dt_depth_{depth}_metrics.txt'), 'w') as f:
        f.write(
            f'Depth: {depth}\nAccuracy: {acc:.4f}\nF1: {f1:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n')

# Wykres podsumowujący
plt.figure(figsize=(10, 6))
plt.plot(max_depths, accuracies, marker='o', label='Accuracy')
plt.plot(max_depths, f1s, marker='o', label='F1 Score')
plt.plot(max_depths, recalls, marker='o', label='Recall')
plt.plot(max_depths, precisions, marker='o', label='Precision')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Score')
plt.title('Decision Tree Performance vs. Max Depth (all features)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'decision_tree_depth_comparison.png'))
plt.close()

print('Eksperyment z głębokością drzewa zakończony. Wyniki i wykres zapisane w', RESULTS_DIR)
