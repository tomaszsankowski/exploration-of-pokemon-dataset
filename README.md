# Data Exploration Project  
**Topic:** Pokémon – Weedle's Cave  
**Stage 1:** Problem Understanding + Data Understanding  
**Authors:**  
- Tomasz Sankowski – 193363  
- Piotr Sulewski – 192594  
- Michał Konieczny – 188667  
**Date:** 2025-05-28  

---

## 📌 General Description of the Dataset

The "Weedle's Cave" dataset contains information about battles between Pokémon characters and their attributes. The data originates from Kaggle and is designed for educational purposes. The task is to build a predictive model that can forecast the outcome of battles between two Pokémon based on their features.

---

## 🎯 Exploration Goal and Success Criteria

The goal of the project is binary classification – to predict whether the first Pokémon will win the battle.  
**Success criteria:**
- Achieve an accuracy of at least 90% on the validation set,
- Identify the most important features influencing battle outcomes.

---

## 🧾 Dataset Characteristics

### Source:  
https://www.kaggle.com/datasets/terminus7/pokemon-challenge

### Format:  
CSV

### Composition:
- `pokemon.csv` – data about 800 Pokémon (ID, name, types, stats)
- `combats.csv` – 50,000 battles (IDs of two Pokémon, winner ID)
- `tests.csv` – battles without winner label (for performance evaluation only)

---

## 🔢 Attribute Description

### **pokemon.csv**

| Attribute     | Type        | Description |
|---------------|-------------|-------------|
| ID            | Numeric     | Unique Pokémon identifier |
| Name          | Nominal     | Pokémon name |
| Type 1        | Nominal     | Primary type (e.g., Fire, Water) |
| Type 2        | Nominal     | Secondary type (optional) |
| HP            | Numeric     | Health Points |
| Attack        | Numeric     | Physical attack strength |
| Defense       | Numeric     | Physical defense strength |
| Sp. Atk       | Numeric     | Special attack strength |
| Sp. Def       | Numeric     | Special defense strength |
| Speed         | Numeric     | Speed attribute |
| Generation    | Numeric     | Generation number (1–6) |
| Legendary     | Nominal     | Indicates if the Pokémon is legendary (True/False) |

### **combats.csv**

| Attribute       | Type      | Description |
|-----------------|-----------|-------------|
| First_pokemon   | Numeric   | ID of the first Pokémon in battle (attacks first) |
| Second_pokemon  | Numeric   | ID of the second Pokémon in battle |
| Winner          | Numeric   | ID of the winning Pokémon |

---

## 📊 Exploratory Data Analysis (EDA) Results

### Value Distributions:
- Histograms for all numeric features (`HP`, `Attack`, `Defense`, etc.).
- Histograms for categorical features such as Pokémon types and Legendary status.
- Distribution of wins: how often the first Pokémon wins versus the second.

### Correlations:
- Correlation analysis between numerical attributes (e.g., `Attack` vs `Defense`).
- Correlation heatmap of numeric features.

### Preliminary Findings:
- Pokémon types are unevenly distributed.
- Legendary Pokémon are rare but have significantly higher stats.

---

## ❗ Data Quality Notes

- **Missing Data:** Only missing values in `Type 2` (no secondary type).
- **Outliers:** Legendary Pokémon are outliers with very high stats.
- **Inconsistencies:** None found.
- **Unclear Data:** None.

---