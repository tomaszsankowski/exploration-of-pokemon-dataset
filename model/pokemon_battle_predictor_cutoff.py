import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import os
import time
import matplotlib.pyplot as plt
from itertools import combinations
import csv

# Folder na wyniki
RESULTS_DIR = 'model/supervised_cutoff/'
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

# Ucinamy zbiór treningowy do 30% oryginalnego rozmiaru
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train_cut, _, y_train_cut, _ = train_test_split(
    X_train, y_train, test_size=0.7, random_state=42)

model_names = []
accuracies = []
f1s = []
recalls = []
specificities = []
train_times = []
predict_times = []

# Funkcja do obliczania swoistości


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float('nan')


# Random Forest
start_train = time.time()
rf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_cut, y_train_cut)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_rf = rf.predict(X_val)
predict_times.append(time.time() - start_pred)
acc_rf = accuracy_score(y_val, y_pred_rf)
f1_rf = f1_score(y_val, y_pred_rf, average='weighted')
recall_rf = recall_score(y_val, y_pred_rf)
spec_rf = specificity_score(y_val, y_pred_rf)
model_names.append('RandomForest')
accuracies.append(acc_rf)
f1s.append(f1_rf)
recalls.append(recall_rf)
specificities.append(spec_rf)
with open(RESULTS_DIR + 'rf_metrics.txt', 'w') as f:
    f.write(
        f'Accuracy: {acc_rf:.4f}\nF1: {f1_rf:.4f}\nSensitivity: {recall_rf:.4f}\nSpecificity: {spec_rf:.4f}\n')

# Logistic Regression
start_train = time.time()
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_cut, y_train_cut)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_logreg = logreg.predict(X_val)
predict_times.append(time.time() - start_pred)
acc_logreg = accuracy_score(y_val, y_pred_logreg)
f1_logreg = f1_score(y_val, y_pred_logreg, average='weighted')
recall_logreg = recall_score(y_val, y_pred_logreg)
spec_logreg = specificity_score(y_val, y_pred_logreg)
model_names.append('LogisticRegression')
accuracies.append(acc_logreg)
f1s.append(f1_logreg)
recalls.append(recall_logreg)
specificities.append(spec_logreg)
with open(RESULTS_DIR + 'logreg_metrics.txt', 'w') as f:
    f.write(
        f'Accuracy: {acc_logreg:.4f}\nF1: {f1_logreg:.4f}\nSensitivity: {recall_logreg:.4f}\nSpecificity: {spec_logreg:.4f}\n')

# SVM
start_train = time.time()
svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train_cut, y_train_cut)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_svc = svc.predict(X_val)
predict_times.append(time.time() - start_pred)
acc_svc = accuracy_score(y_val, y_pred_svc)
f1_svc = f1_score(y_val, y_pred_svc, average='weighted')
recall_svc = recall_score(y_val, y_pred_svc)
spec_svc = specificity_score(y_val, y_pred_svc)
model_names.append('SVM')
accuracies.append(acc_svc)
f1s.append(f1_svc)
recalls.append(recall_svc)
specificities.append(spec_svc)
with open(RESULTS_DIR + 'svc_metrics.txt', 'w') as f:
    f.write(
        f'Accuracy: {acc_svc:.4f}\nF1: {f1_svc:.4f}\nSensitivity: {recall_svc:.4f}\nSpecificity: {spec_svc:.4f}\n')

# Decision Tree
start_train = time.time()
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train_cut, y_train_cut)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_dt = dt.predict(X_val)
predict_times.append(time.time() - start_pred)
acc_dt = accuracy_score(y_val, y_pred_dt)
f1_dt = f1_score(y_val, y_pred_dt, average='weighted')
recall_dt = recall_score(y_val, y_pred_dt)
spec_dt = specificity_score(y_val, y_pred_dt)
model_names.append('DecisionTree')
accuracies.append(acc_dt)
f1s.append(f1_dt)
recalls.append(recall_dt)
specificities.append(spec_dt)
with open(RESULTS_DIR + 'dt_metrics.txt', 'w') as f:
    f.write(
        f'Accuracy: {acc_dt:.4f}\nF1: {f1_dt:.4f}\nSensitivity: {recall_dt:.4f}\nSpecificity: {spec_dt:.4f}\n')

# --- EKSPERYMENTY POJEDYNCZYCH CECH ---

# Lista cech do testowania pojedynczo i parami
single_features = stats_columns + ['Legendary_P1', 'Legendary_P2']
pair_features = list(combinations(stats_columns, 2))

feature_exp_results = []

# Testuj pojedyncze cechy
for feat in single_features:
    feats = [f'Diff_{feat}'] if feat in stats_columns else [feat]
    X_feat = merged_df[feats].fillna(0)
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        X_feat, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_f, y_train_f)
    y_pred_f = rf.predict(X_val_f)
    acc = accuracy_score(y_val_f, y_pred_f)
    f1 = f1_score(y_val_f, y_pred_f, average='weighted')
    recall = recall_score(y_val_f, y_pred_f)
    spec = specificity_score(y_val_f, y_pred_f)
    feature_exp_results.append(
        {'features': feats, 'accuracy': acc, 'f1': f1, 'recall': recall, 'specificity': spec})

# Testuj pary cech
for f1_, f2_ in pair_features:
    feats = [f'Diff_{f1_}', f'Diff_{f2_}']
    X_feat = merged_df[feats].fillna(0)
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        X_feat, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_f, y_train_f)
    y_pred_f = rf.predict(X_val_f)
    acc = accuracy_score(y_val_f, y_pred_f)
    f1 = f1_score(y_val_f, y_pred_f, average='weighted')
    recall = recall_score(y_val_f, y_pred_f)
    spec = specificity_score(y_val_f, y_pred_f)
    feature_exp_results.append(
        {'features': feats, 'accuracy': acc, 'f1': f1, 'recall': recall, 'specificity': spec})

# Zapisz wyniki eksperymentów do CSV
with open(RESULTS_DIR + 'feature_importance_experiments.csv', 'w', newline='') as csvfile:
    fieldnames = ['features', 'accuracy', 'f1', 'recall', 'specificity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in feature_exp_results:
        writer.writerow({
            'features': ','.join(row['features']),
            'accuracy': row['accuracy'],
            'f1': row['f1'],
            'recall': row['recall'],
            'specificity': row['specificity']
        })

# --- WYKRESY ---
plt.figure(figsize=(8, 4))
bars = plt.bar(model_names, recalls, color='blue')
plt.ylabel('Sensitivity (Recall)')
plt.title('Sensitivity (Recall) of Models')
for bar, val in zip(bars, recalls):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom')
plt.savefig(RESULTS_DIR + 'model_sensitivity.png')
plt.close()

plt.figure(figsize=(8, 4))
bars = plt.bar(model_names, specificities, color='purple')
plt.ylabel('Specificity')
plt.title('Specificity of Models')
for bar, val in zip(bars, specificities):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom')
plt.savefig(RESULTS_DIR + 'model_specificity.png')
plt.close()

plt.figure(figsize=(8, 4))
bars = plt.bar(model_names, accuracies, color='skyblue')
plt.ylabel('Accuracy')
plt.title('Accuracy of Models')
for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom')
plt.savefig(RESULTS_DIR + 'model_accuracies.png')
plt.close()

plt.figure(figsize=(8, 4))
bars = plt.bar(model_names, f1s, color='orange')
plt.ylabel('F1 Score')
plt.title('F1 Score of Models')
for bar, val in zip(bars, f1s):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom')
plt.savefig(RESULTS_DIR + 'model_f1s.png')
plt.close()

plt.figure(figsize=(8, 4))
bars = plt.bar(model_names, train_times, color='green')
plt.ylabel('Training Time (s)')
plt.title('Training Time of Models')
for bar, val in zip(bars, train_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom')
plt.savefig(RESULTS_DIR + 'model_train_times.png')
plt.close()

plt.figure(figsize=(8, 4))
bars = plt.bar(model_names, predict_times, color='red')
plt.ylabel('Prediction Time (s)')
plt.title('Prediction Time of Models')
for bar, val in zip(bars, predict_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom')
plt.savefig(RESULTS_DIR + 'model_predict_times.png')
plt.close()

print('Wyniki cutoff zapisane w', RESULTS_DIR)
