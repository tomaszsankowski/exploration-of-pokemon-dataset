import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

RESULTS_DIR = 'model/supervised_cutoff/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Wczytaj dane
csv_path = os.path.join(RESULTS_DIR, 'feature_importance_experiments.csv')
df = pd.read_csv(csv_path)

# Zamień stringi z listami cech na czytelne etykiety


def parse_features(s):
    # Usuwa cudzysłowy i przecinki, zamienia na czytelny format
    if ',' in s:
        return '+'.join([x.strip() for x in s.replace('"', '').split(',')])
    return s.replace('"', '')


df['features_label'] = df['features'].apply(parse_features)

# Wykres: accuracy dla każdej cechy i pary cech
plt.figure(figsize=(12, 6))
plt.bar(df['features_label'], df['accuracy'], color='skyblue')
plt.ylabel('Accuracy')
plt.title('Accuracy for Single Features and Feature Pairs')
plt.xticks(rotation=90)
for i, v in enumerate(df['accuracy']):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_accuracy.png'))
plt.close()

# Wykres: F1 dla każdej cechy i pary cech
plt.figure(figsize=(12, 6))
plt.bar(df['features_label'], df['f1'], color='orange')
plt.ylabel('F1 Score')
plt.title('F1 Score for Single Features and Feature Pairs')
plt.xticks(rotation=90)
for i, v in enumerate(df['f1']):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_f1.png'))
plt.close()

print('Wykresy accuracy i F1 dla cech zapisane w', RESULTS_DIR)
