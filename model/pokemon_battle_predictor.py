import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from joblib import dump
import os
# --- EKSPERYMENTY ---
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time
import matplotlib.pyplot as plt

# Zmienna na folder do zapisu wyników
RESULTS_DIR = 'model/supervised/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load datasets
pokemon_df = pd.read_csv(
    'c:\\\\Users\\\\piotr\\\\VsCodeProjects\\\\exploration-of-pokemon-dataset\\\\pokemon.csv')
combats_df = pd.read_csv(
    'c:\\\\Users\\\\piotr\\\\VsCodeProjects\\\\exploration-of-pokemon-dataset\\\\combats.csv')
tests_df = pd.read_csv(
    'c:\\\\Users\\\\piotr\\\\VsCodeProjects\\\\exploration-of-pokemon-dataset\\\\tests.csv')

# 1. Data Preparation (Feature Engineering)

# IMPORTANT: Define the name of the ID column in your 'pokemon.csv' file.
# The most common name is '#'. If your CSV uses a different name (e.g., 'ID', 'Number'),
# you MUST change this variable to match your CSV file's header for the Pokemon ID.
# If this is incorrect, you will likely get a KeyError during the merge operations below
# (e.g., KeyError: '#_P1' if '#' is not a column in your pokemon.csv).
ID_COLUMN_NAME_IN_POKEMON_CSV = 'ID'

# Merge combat data with Pokémon stats for First_pokemon
# Create a version of pokemon_df with all columns suffixed with _P1
pokemon_P1_stats = pokemon_df.add_suffix('_P1')
# The ID column for P1 is now ID_COLUMN_NAME_IN_POKEMON_CSV + '_P1'
merged_df = pd.merge(combats_df, pokemon_P1_stats,
                     left_on='First_pokemon',
                     right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P1')

# Merge again for Second_pokemon
# Create a version of pokemon_df with all columns suffixed with _P2
pokemon_P2_stats = pokemon_df.add_suffix('_P2')
# The ID column for P2 is now ID_COLUMN_NAME_IN_POKEMON_CSV + '_P2'
merged_df = pd.merge(merged_df, pokemon_P2_stats,
                     left_on='Second_pokemon',
                     right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P2')

# Now, columns from the first Pokemon have _P1 suffix (e.g., HP_P1, Attack_P1)
# and columns from the second Pokemon have _P2 suffix (e.g., HP_P2, Attack_P2).
# This makes subsequent feature engineering (like Diff_HP = HP_P1 - HP_P2) correct.
# The original suffixes=('_P1', '_P2') in pd.merge was not achieving this for all columns.

# Rename columns for clarity after merge
# P1 refers to First_pokemon, P2 to Second_pokemon
# Example: 'HP_P1' is HP of First_pokemon, 'HP_P2' is HP of Second_pokemon
# Adjust column names based on your actual merged DataFrame structure
# For instance, if after the first merge, HP becomes 'HP', and after the second, it's 'HP_y',
# you'll need to rename them appropriately.
# Assuming the suffixes correctly differentiate them:
# e.g., 'Name_P1', 'Type 1_P1', 'HP_P1', etc. for First_pokemon
# and 'Name_P2', 'Type 1_P2', 'HP_P2', etc. for Second_pokemon

# Create new features: Differences in stats
stats_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
for stat in stats_columns:
    merged_df[f'Diff_{stat}'] = merged_df[f'{stat}_P1'] - \
        merged_df[f'{stat}_P2']

# Create new features: Binary legendary status
merged_df['Legendary_P1'] = merged_df['Legendary_P1'].astype(int)
merged_df['Legendary_P2'] = merged_df['Legendary_P2'].astype(int)

# TODO: Implement Type Advantage Feature
# This requires a type effectiveness matrix. For simplicity, this is skipped here.
# Example: merged_df['Type_Advantage'] = calculate_type_advantage(merged_df['Type 1_P1'], merged_df['Type 2_P1'], merged_df['Type 1_P2'], merged_df['Type 2_P2'])


# Transform target variable: 0 if First_pokemon won, 1 if Second_pokemon won
merged_df['Winner_Binary'] = merged_df.apply(
    lambda row: 0 if row['Winner'] == row['First_pokemon'] else 1, axis=1)

# Select features for the model
# Include engineered features and potentially others like 'Legendary_P1', 'Legendary_P2'
# Exclude identifiers like 'First_pokemon', 'Second_pokemon', 'Winner', '#_P1', '#_P2', 'Name_P1', 'Name_P2', etc.
# Also exclude original stat columns if only differences are used, or include them if deemed useful.
features = [f'Diff_{stat}' for stat in stats_columns] + \
    ['Legendary_P1', 'Legendary_P2']
# Add other relevant features from pokemon_df if needed, ensuring they are present and correctly named in merged_df
# For example, if you want to include the base stats directly:
# features += [f'{stat}_P1' for stat in stats_columns] + [f'{stat}_P2' for stat in stats_columns]


X = merged_df[features]
y = merged_df['Winner_Binary']

# Handle missing values (if any) - e.g., by imputation
X = X.fillna(X.mean())


# 2. Data Splitting
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- CZASY I WYNIKI MODELI ---
model_names = []
accuracies = []
f1s = []
train_times = []
predict_times = []

# 3. Model Selection and Training
# Random Forest
start_train = time.time()
model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_val = model.predict(X_val)
predict_times.append(time.time() - start_pred)
accuracy = accuracy_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val, average='weighted')
model_names.append('RandomForest')
accuracies.append(accuracy)
f1s.append(f1)

dump(model, RESULTS_DIR + 'pokemon_rf_model.joblib')
print('Model zapisany do', RESULTS_DIR + 'pokemon_rf_model.joblib')

# 4. Model Evaluation
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
# Use weighted for multiclass/imbalanced binary
precision = precision_score(y_val, y_pred_val, average='weighted')
# Use weighted for multiclass/imbalanced binary
f1 = f1_score(y_val, y_pred_val, average='weighted')

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation F1-score: {f1:.4f}")

# --- EKSPERYMENTY ---
# 1. Eksperyment: tylko różnice statystyk (bez cech legendarnych)
features_exp1 = [f'Diff_{stat}' for stat in stats_columns]
X_exp1 = merged_df[features_exp1].fillna(0)
model_exp1 = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
model_exp1.fit(X_train[features_exp1], y_train)
y_pred_exp1 = model_exp1.predict(X_val[features_exp1])
acc_exp1 = accuracy_score(y_val, y_pred_exp1)
f1_exp1 = f1_score(y_val, y_pred_exp1, average='weighted')
with open(RESULTS_DIR + 'exp1_metrics.txt', 'w') as f:
    f.write(
        f'Accuracy (only stat diffs): {acc_exp1:.4f}\nF1 (only stat diffs): {f1_exp1:.4f}\n')

# 2. Eksperyment: Standaryzacja cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
model_scaled = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_val_scaled)
acc_scaled = accuracy_score(y_val, y_pred_scaled)
f1_scaled = f1_score(y_val, y_pred_scaled, average='weighted')
with open(RESULTS_DIR + 'exp2_metrics.txt', 'w') as f:
    f.write(
        f'Accuracy (standardized): {acc_scaled:.4f}\nF1 (standardized): {f1_scaled:.4f}\n')

# 3. Eksperyment: Inne parametry modelu
model_deep = RandomForestClassifier(
    n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')
model_deep.fit(X_train, y_train)
y_pred_deep = model_deep.predict(X_val)
acc_deep = accuracy_score(y_val, y_pred_deep)
f1_deep = f1_score(y_val, y_pred_deep, average='weighted')
with open(RESULTS_DIR + 'exp3_metrics.txt', 'w') as f:
    f.write(
        f'Accuracy (n_estimators=200, max_depth=20): {acc_deep:.4f}\nF1: {f1_deep:.4f}\n')

# 4. Model: Logistic Regression
start_train = time.time()
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_logreg = logreg.predict(X_val)
predict_times.append(time.time() - start_pred)
acc_logreg = accuracy_score(y_val, y_pred_logreg)
f1_logreg = f1_score(y_val, y_pred_logreg, average='weighted')
model_names.append('LogisticRegression')
accuracies.append(acc_logreg)
f1s.append(f1_logreg)
with open(RESULTS_DIR + 'logreg_metrics.txt', 'w') as f:
    f.write(f'Logistic Regression Accuracy: {acc_logreg:.4f}\n')
    f.write(f'Logistic Regression F1: {f1_logreg:.4f}\n')
print(f'Logistic Regression Accuracy: {acc_logreg:.4f}, F1: {f1_logreg:.4f}')

# 5. Model: SVM
start_train = time.time()
svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train, y_train)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_svc = svc.predict(X_val)
predict_times.append(time.time() - start_pred)
acc_svc = accuracy_score(y_val, y_pred_svc)
f1_svc = f1_score(y_val, y_pred_svc, average='weighted')
model_names.append('SVM')
accuracies.append(acc_svc)
f1s.append(f1_svc)
with open(RESULTS_DIR + 'svc_metrics.txt', 'w') as f:
    f.write(f'SVM Accuracy: {acc_svc:.4f}\n')
    f.write(f'SVM F1: {f1_svc:.4f}\n')
print(f'SVM Accuracy: {acc_svc:.4f}, F1: {f1_svc:.4f}')

# 6. Model: Decision Tree
start_train = time.time()
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
train_times.append(time.time() - start_train)
start_pred = time.time()
y_pred_dt = dt.predict(X_val)
predict_times.append(time.time() - start_pred)
acc_dt = accuracy_score(y_val, y_pred_dt)
f1_dt = f1_score(y_val, y_pred_dt, average='weighted')
model_names.append('DecisionTree')
accuracies.append(acc_dt)
f1s.append(f1_dt)
with open(RESULTS_DIR + 'dt_metrics.txt', 'w') as f:
    f.write(f'Decision Tree Accuracy: {acc_dt:.4f}\n')
    f.write(f'Decision Tree F1: {f1_dt:.4f}\n')
print(f'Decision Tree Accuracy: {acc_dt:.4f}, F1: {f1_dt:.4f}')

# --- WYKRESY ---
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

# 6. Hyperparameter Tuning (Optional)
# Example using GridSearchCV (can be time-consuming)
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# print(f"Best parameters: {grid_search.best_params_}")
# model = grid_search.best_estimator_ # Use the best model found

# 7. Prediction on Test Set
# Prepare tests_df similar to how combats_df was prepared
# IMPORTANT: The same ID_COLUMN_NAME_IN_POKEMON_CSV defined earlier is used here.
test_pokemon_P1_stats = pokemon_df.add_suffix('_P1')
test_merged_df = pd.merge(
    tests_df, test_pokemon_P1_stats, left_on='First_pokemon', right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P1')

test_pokemon_P2_stats = pokemon_df.add_suffix('_P2')
test_merged_df = pd.merge(test_merged_df, test_pokemon_P2_stats,
                          left_on='Second_pokemon', right_on=ID_COLUMN_NAME_IN_POKEMON_CSV + '_P2')

# Create the same engineered features for the test set
for stat in stats_columns:
    test_merged_df[f'Diff_{stat}'] = test_merged_df[f'{stat}_P1'] - \
        test_merged_df[f'{stat}_P2']
test_merged_df['Legendary_P1'] = test_merged_df['Legendary_P1'].astype(int)
test_merged_df['Legendary_P2'] = test_merged_df['Legendary_P2'].astype(int)
# TODO: Add Type Advantage for test_merged_df as well

X_test = test_merged_df[features]
X_test = X_test.fillna(X_test.mean())  # Handle missing values in test set

# Predict winners
test_predictions_binary = model.predict(X_test)

# Convert binary predictions back to Pokémon numbers/IDs
# 0 means First_pokemon wins, 1 means Second_pokemon wins
predicted_winners = test_merged_df.apply(
    lambda row: row['First_pokemon'] if test_predictions_binary[row.name] == 0 else row['Second_pokemon'],
    axis=1
)

# Create submission file (if required, e.g., for a competition)
submission_df = pd.DataFrame({
    'First_pokemon': test_merged_df['First_pokemon'],
    'Second_pokemon': test_merged_df['Second_pokemon'],
    'Winner': predicted_winners
})
submission_df.to_csv(
    RESULTS_DIR + 'predictions.csv', index=False)

print("Predictions saved to", RESULTS_DIR + 'predictions.csv')

# Further improvements:
# - More sophisticated feature engineering (e.g., type effectiveness, more interaction terms).
# - Trying other models like XGBoost or LightGBM.
# - More thorough hyperparameter tuning.
# - Cross-validation strategy that respects potential time-series nature if applicable (though not obvious here).
# - Handling of categorical features (Type 1, Type 2) more explicitly, e.g., via one-hot encoding or embedding, if not using tree-based models that can handle them natively or if type advantage isn't comprehensive.
# - Ensure column names after merges are handled robustly. The current suffixes '_P1', '_P2' are assumed but might need adjustment based on actual pandas behavior with your specific column names (e.g., if '#' is the merge key, it might become '#_x', '#_y').
