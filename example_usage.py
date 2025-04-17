# -*- coding: utf-8 -*-
"""
Örnek Kullanım Dosyası

Bu dosya, data_preprocessor.py ve model_selector.py modüllerinin birlikte nasıl kullanılacağını gösterir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris, make_blobs

# Kendi modüllerimizi import et
from data_preprocessor import DataPreprocessor
from model_selector import ModelSelector

# Görselleştirme ayarları
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Regresyon Örneği
print("\n" + "="*50)
print("REGRESYON ÖRNEĞİ")
print("="*50)

# California ev fiyatları veri setini yükle
housing = fetch_california_housing()
california_data = pd.DataFrame(housing.data, columns=housing.feature_names)
california_data['PRICE'] = housing.target

print("\nDataset size:", california_data.shape)
print("\nFirst 5 rows:")
print(california_data.head())

# Data preprocessing
print("\n1. Data Preprocessing Step")
preprocessor = DataPreprocessor(verbose=True)

# Define preprocessing steps
preprocessing_steps = {
    'handle_missing_values': {'method': 'mean'},
    'handle_outliers': {'method': 'clip', 'threshold': 1.5},
    'scale_features': {'method': 'standard'},
    'feature_selection': {'method': 'importance', 'k': 8}
}

# Process data
processed_data = preprocessor.fit_transform(
    data=california_data,
    target='PRICE',
    preprocessing_steps=preprocessing_steps
)

print("\nProcessed dataset size:", processed_data.shape)
print("\nFirst 5 rows of processed data:")
print(processed_data.head())

# Separate target variable
X = processed_data.drop(columns=['PRICE'])
y = processed_data['PRICE']

# Model selection
print("\n2. Model Selection Step")
model_selector = ModelSelector(problem_type='regression')

# Add ensemble models
model_selector.add_ensemble_model('stacking')
model_selector.add_ensemble_model('voting')

# Train models
model_selector.fit(X, y)

# Get the best model
best_model_info = model_selector.get_best_model()

print(f"\nBest regression model: {best_model_info['model_name']}")
print(f"R2 skoru: {best_model_info['score']:.4f}")
print(f"Model parametreleri: {best_model_info['parameters']}")

# Tüm model sonuçlarını göster
results = model_selector.get_results()
print("\nAll model results:")
for model_name, result in results.items():
    if 'error' in result:
        print(f"{model_name}: Hata - {result['error']}")
    else:
        print(f"{model_name}: R2={result['r2']:.4f}, RMSE={result['rmse']:.4f}, Eğitim süresi={result['train_time']:.4f}s")

# Model açıklanabilirliği (SHAP değerleri)
print("\n3. Model Açıklanabilirliği (SHAP)")
print("\nEn iyi model için SHAP değerleri:")
# En iyi model için SHAP değerlerini hesapla ve görselleştir
shap_values, explainer = model_selector.explain_model(
    X=X.iloc[:100],  # Hesaplama süresini azaltmak için ilk 100 örnek
    plot_type='bar',  # Bar plot (özellik önem sıralaması)
    max_display=10    # En önemli 10 özelliği göster
)

# Farklı görselleştirme tipleri için örnekler
print("\nFarklı görselleştirme tipleri (en iyi model için):")
# Beeswarm plot (özellik dağılımları)
model_selector.explain_model(
    X=X.iloc[:100],
    plot_type='beeswarm',
    max_display=10
)

# Waterfall plot (tek bir örnek için tahmin açıklaması)
model_selector.explain_model(
    X=X.iloc[:1],  # Sadece ilk örnek
    plot_type='waterfall',
    max_display=10
)

# SHAP değerlerini dosyaya kaydetme örneği
print("\nSHAP grafiğini dosyaya kaydetme:")
model_selector.explain_model(
    X=X.iloc[:100],
    plot_type='bar',
    max_display=10,
    save_plot=True,
    filename='shap_values_regression.png'
)

# 2. Sınıflandırma Örneği
print("\n" + "="*50)
print("SINIFLANDIRMA ÖRNEĞİ")
print("="*50)

# Iris veri setini yükle
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

print("\nVeri seti boyutu:", iris_data.shape)
print("\nFirst 5 rows:")
print(iris_data.head())

# Data preprocessing
print("\n1. Data Preprocessing Step")
preprocessor = DataPreprocessor(verbose=True)

# Define preprocessing steps
preprocessing_steps = {
    'scale_features': {'method': 'standard'},
    'feature_selection': {'method': 'kbest', 'k': 3}
}

# Process data
processed_data = preprocessor.fit_transform(
    data=iris_data,
    target='target',
    preprocessing_steps=preprocessing_steps
)

print("\nProcessed dataset size:", processed_data.shape)
print("\nFirst 5 rows of processed data:")
print(processed_data.head())

# Separate target variable
X = processed_data.drop(columns=['target'])
y = processed_data['target']

# Model selection
print("\n2. Model Selection Step")
model_selector = ModelSelector(problem_type='classification')

# Add ensemble models
model_selector.add_ensemble_model('stacking')
model_selector.add_ensemble_model('voting')

# Train models
model_selector.fit(X, y)

# Get the best model
best_model_info = model_selector.get_best_model()

print(f"\nEn iyi sınıflandırma modeli: {best_model_info['model_name']}")
print(f"Doğruluk: {best_model_info['score']:.4f}")
print(f"Model parametreleri: {best_model_info['parameters']}")

# Model açıklanabilirliği (SHAP değerleri)
print("\n3. Model Açıklanabilirliği (SHAP)")
print("\nEn iyi sınıflandırma modeli için SHAP değerleri:")
# En iyi model için SHAP değerlerini hesapla ve görselleştir
shap_values, explainer = model_selector.explain_model(
    X=X,  # Sınıflandırma veri seti
    plot_type='bar',  # Bar plot (özellik önem sıralaması)
    max_display=4     # Tüm özellikleri göster (iris veri seti için 4 özellik var)
)

# Farklı görselleştirme tipleri için örnekler
print("\nFarklı görselleştirme tipleri (en iyi sınıflandırma modeli için):")
# Beeswarm plot (özellik dağılımları)
model_selector.explain_model(
    X=X,
    plot_type='beeswarm',
    max_display=4
)

# Force plot (tek bir örnek için tahmin açıklaması)
model_selector.explain_model(
    X=X.iloc[:1],  # Sadece ilk örnek
    plot_type='force',
    max_display=4
)

# SHAP değerlerini dosyaya kaydetme örneği
print("\nSHAP grafiğini dosyaya kaydetme:")
model_selector.explain_model(
    X=X,
    plot_type='bar',
    max_display=4,
    save_plot=True,
    filename='shap_values_classification.png'
)

# 3. Kümeleme Örneği
print("\n" + "="*50)
print("KÜMELEME ÖRNEĞİ")
print("="*50)

# Yapay kümeleme verisi oluştur
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
cluster_data = pd.DataFrame(X, columns=['feature1', 'feature2'])

print("\nVeri seti boyutu:", cluster_data.shape)
print("\nFirst 5 rows:")
print(cluster_data.head())

# Data preprocessing
print("\n1. Data Preprocessing Step")
preprocessor = DataPreprocessor(verbose=True)

# Define preprocessing steps
preprocessing_steps = {
    'scale_features': {'method': 'standard'}
}

# Process data
processed_data = preprocessor.fit_transform(
    data=cluster_data,
    preprocessing_steps=preprocessing_steps
)

print("\nProcessed dataset size:", processed_data.shape)
print("\nFirst 5 rows of processed data:")
print(processed_data.head())

# Model selection
print("\n2. Model Selection Step")
model_selector = ModelSelector(problem_type='clustering')

# Train models
model_selector.fit(processed_data, None)

# Get the best model
best_model_info = model_selector.get_best_model()

print(f"\nEn iyi kümeleme modeli: {best_model_info['model_name']}")
print(f"Eğitim süresi: {best_model_info['score']:.4f} saniye")
print(f"Model parametreleri: {best_model_info['parameters']}")

# Model açıklanabilirliği (SHAP değerleri)
print("\n3. Model Açıklanabilirliği (SHAP)")
print("\nNot: SHAP açıklamaları kümeleme modelleri için desteklenmemektedir.")
print("Kümeleme modelleri için model açıklanabilirliği, kümeleme sonuçlarının görselleştirilmesi")
print("veya küme merkezlerinin incelenmesi gibi farklı yöntemlerle sağlanabilir.")

# 4. EDA Örnekleri
print("\n" + "="*50)
print("EDA ÖRNEKLERİ")
print("="*50)

# California veri seti üzerinde EDA
print("\nCalifornia veri seti üzerinde EDA örnekleri:")
eda_preprocessor = DataPreprocessor(verbose=False)

# Eksik değer analizi
print("\n1. Eksik değer analizi:")
print(california_data.isnull().sum())

# Aykırı değer analizi
print("\n2. Aykırı değer analizi:")
eda_preprocessor.original_data = california_data
# Aykırı değerleri görselleştirmek için bu satırı aktif edin:
# eda_preprocessor.plot_outliers(columns=['MedInc', 'HouseAge', 'AveRooms', 'PRICE'])

# Korelasyon matrisi
print("\n3. Korelasyon matrisi:")
corr = california_data.corr()
print(corr['PRICE'].sort_values(ascending=False))
# Korelasyon matrisini görselleştirmek için bu satırı aktif edin:
# eda_preprocessor.plot_correlation_matrix()

# Özellik önemleri
print("\n4. Özellik önemleri:")
# Özellik önemlerini görselleştirmek için bu satırı aktif edin:
# eda_preprocessor.plot_feature_importance(target='PRICE')

print("\nExample usage completed.")