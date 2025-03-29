# -*- coding: utf-8 -*-
"""
Model Selector Test File - Regression

This file is used to test the regression functionality of the ModelSelector class in
model_selector.py. Using the California Housing dataset from scikit-learn, it finds
the best regression model and its parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from model_selector import ModelSelector

# Load California Housing dataset
print("\n=== California Housing Veri Seti Yükleniyor ===\n")
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# Display dataset information
print(f"Veri seti boyutu: {X.shape}")
print(f"Özellikler: {feature_names}")
print(f"Hedef değişken: {housing.target_names[0]}")
print(f"Hedef değişken min: {y.min():.2f}, max: {y.max():.2f}, ortalama: {y.mean():.2f}")

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize the dataset
plt.figure(figsize=(12, 10))

# Distribution of target variable
plt.subplot(2, 2, 1)
sns.histplot(y, kde=True)
plt.title('Hedef Değişken Dağılımı (Ev Fiyatları)')
plt.xlabel('Ev Fiyatı (100,000$)')

# Correlation between features
plt.subplot(2, 2, 2)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Özellikler Arası Korelasyon')

# Test regression models
print("\n=== Regresyon Modelleri Testi Başlıyor ===\n")
ms_reg = ModelSelector('regression')

# Train and evaluate models
print("Modeller eğitiliyor ve değerlendiriliyor...")
ms_reg.fit(X_scaled, y, test_size=0.3, random_state=42)

# Display results
print("\n=== Model Sonuçları ===\n")
results = ms_reg.get_results()

# Show results in table format
r2_scores = {}
rmse_scores = {}
train_times = {}

for model_name, result in results.items():
    if 'error' in result:
        print(f"{model_name}: HATA - {result['error']}")
        continue
    
    r2 = result['r2']
    rmse = result['rmse']
    train_time = result['train_time']
    
    r2_scores[model_name] = r2
    rmse_scores[model_name] = rmse
    train_times[model_name] = train_time
    
    print(f"{model_name}:")
    print(f"  R² Skoru: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Eğitim Süresi: {train_time:.4f} saniye")
    print()

# Show the best model
best_model_info = ms_reg.get_best_model()
print("\n=== En İyi Model ===\n")
print(f"Model: {best_model_info['model_name']}")
print(f"R² Skoru: {best_model_info['score']:.4f}")
print("\nParametreler:")
for param, value in best_model_info['parameters'].items():
    print(f"  {param}: {value}")

# Visualize R² scores of models
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sorted_r2 = {k: v for k, v in sorted(r2_scores.items(), key=lambda item: item[1], reverse=True)}
plt.bar(sorted_r2.keys(), sorted_r2.values(), color='skyblue')
plt.xticks(rotation=90)
plt.title('Model R² Skorları')
plt.ylabel('R² Skoru')
plt.ylim(0, 1.0)  # R² score should be between 0-1

# Visualize RMSE scores of models
plt.subplot(2, 2, 2)
sorted_rmse = {k: v for k, v in sorted(rmse_scores.items(), key=lambda item: item[1])}
plt.bar(sorted_rmse.keys(), sorted_rmse.values(), color='lightgreen')
plt.xticks(rotation=90)
plt.title('Model RMSE Değerleri')
plt.ylabel('RMSE')

# Visualize training times of models
plt.subplot(2, 2, 3)
sorted_times = {k: v for k, v in sorted(train_times.items(), key=lambda item: item[1])}
plt.bar(sorted_times.keys(), sorted_times.values(), color='salmon')
plt.xticks(rotation=90)
plt.title('Model Eğitim Süreleri')
plt.ylabel('Süre (saniye)')

# Visualize predictions of the best model
plt.subplot(2, 2, 4)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
y_pred = best_model_info['model'].predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title(f'En İyi Model ({best_model_info["model_name"]}) Tahminleri')

plt.tight_layout()
plt.savefig('model_comparison_regression.png')
plt.show()

print("\nTest başarılı bir şekilde tamamlandı!")
print("Model karşılaştırma grafiği 'model_comparison_regression.png' olarak kaydedildi.")