# -*- coding: utf-8 -*-
"""
Örnek Kullanım Dosyası

Bu dosya, data_preprocessor.py ve model_selector.py modüllerinin birlikte nasıl kullanılacağını gösterir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, load_iris, make_blobs

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

# Boston ev fiyatları veri setini yükle
boston = load_boston()
boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_data['PRICE'] = boston.target

print("\nVeri seti boyutu:", boston_data.shape)
print("\nİlk 5 satır:")
print(boston_data.head())

# Veri ön işleme
print("\n1. Veri Ön İşleme Adımı")
preprocessor = DataPreprocessor(verbose=True)

# Ön işleme adımlarını tanımla
preprocessing_steps = {
    'handle_missing_values': {'method': 'mean'},
    'handle_outliers': {'method': 'clip', 'threshold': 1.5},
    'scale_features': {'method': 'standard'},
    'feature_selection': {'method': 'importance', 'k': 8}
}

# Veriyi işle
processed_data = preprocessor.fit_transform(
    data=boston_data,
    target='PRICE',
    preprocessing_steps=preprocessing_steps
)

print("\nİşlenmiş veri seti boyutu:", processed_data.shape)
print("\nİşlenmiş verinin ilk 5 satırı:")
print(processed_data.head())

# Hedef değişkeni ayır
X = processed_data.drop(columns=['PRICE'])
y = processed_data['PRICE']

# Model seçimi
print("\n2. Model Seçimi Adımı")
model_selector = ModelSelector(problem_type='regression')

# Ensemble modeller ekle
model_selector.add_ensemble_model('stacking')
model_selector.add_ensemble_model('voting')

# Modelleri eğit
model_selector.fit(X, y)

# En iyi modeli al
best_model_info = model_selector.get_best_model()

print(f"\nEn iyi regresyon modeli: {best_model_info['model_name']}")
print(f"R2 skoru: {best_model_info['score']:.4f}")
print(f"Model parametreleri: {best_model_info['parameters']}")

# Tüm model sonuçlarını göster
results = model_selector.get_results()
print("\nTüm model sonuçları:")
for model_name, result in results.items():
    if 'error' in result:
        print(f"{model_name}: Hata - {result['error']}")
    else:
        print(f"{model_name}: R2={result['r2']:.4f}, RMSE={result['rmse']:.4f}, Eğitim süresi={result['train_time']:.4f}s")

# 2. Sınıflandırma Örneği
print("\n" + "="*50)
print("SINIFLANDIRMA ÖRNEĞİ")
print("="*50)

# Iris veri setini yükle
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

print("\nVeri seti boyutu:", iris_data.shape)
print("\nİlk 5 satır:")
print(iris_data.head())

# Veri ön işleme
print("\n1. Veri Ön İşleme Adımı")
preprocessor = DataPreprocessor(verbose=True)

# Ön işleme adımlarını tanımla
preprocessing_steps = {
    'scale_features': {'method': 'standard'},
    'feature_selection': {'method': 'kbest', 'k': 3}
}

# Veriyi işle
processed_data = preprocessor.fit_transform(
    data=iris_data,
    target='target',
    preprocessing_steps=preprocessing_steps
)

print("\nİşlenmiş veri seti boyutu:", processed_data.shape)
print("\nİşlenmiş verinin ilk 5 satırı:")
print(processed_data.head())

# Hedef değişkeni ayır
X = processed_data.drop(columns=['target'])
y = processed_data['target']

# Model seçimi
print("\n2. Model Seçimi Adımı")
model_selector = ModelSelector(problem_type='classification')

# Ensemble modeller ekle
model_selector.add_ensemble_model('stacking')
model_selector.add_ensemble_model('voting')

# Modelleri eğit
model_selector.fit(X, y)

# En iyi modeli al
best_model_info = model_selector.get_best_model()

print(f"\nEn iyi sınıflandırma modeli: {best_model_info['model_name']}")
print(f"Doğruluk: {best_model_info['score']:.4f}")
print(f"Model parametreleri: {best_model_info['parameters']}")

# 3. Kümeleme Örneği
print("\n" + "="*50)
print("KÜMELEME ÖRNEĞİ")
print("="*50)

# Yapay kümeleme verisi oluştur
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
cluster_data = pd.DataFrame(X, columns=['feature1', 'feature2'])

print("\nVeri seti boyutu:", cluster_data.shape)
print("\nİlk 5 satır:")
print(cluster_data.head())

# Veri ön işleme
print("\n1. Veri Ön İşleme Adımı")
preprocessor = DataPreprocessor(verbose=True)

# Ön işleme adımlarını tanımla
preprocessing_steps = {
    'scale_features': {'method': 'standard'}
}

# Veriyi işle
processed_data = preprocessor.fit_transform(
    data=cluster_data,
    preprocessing_steps=preprocessing_steps
)

print("\nİşlenmiş veri seti boyutu:", processed_data.shape)
print("\nİşlenmiş verinin ilk 5 satırı:")
print(processed_data.head())

# Model seçimi
print("\n2. Model Seçimi Adımı")
model_selector = ModelSelector(problem_type='clustering')

# Modelleri eğit
model_selector.fit(processed_data, None)

# En iyi modeli al
best_model_info = model_selector.get_best_model()

print(f"\nEn iyi kümeleme modeli: {best_model_info['model_name']}")
print(f"Eğitim süresi: {best_model_info['score']:.4f} saniye")
print(f"Model parametreleri: {best_model_info['parameters']}")

# 4. EDA Örnekleri
print("\n" + "="*50)
print("EDA ÖRNEKLERİ")
print("="*50)

# Boston veri seti üzerinde EDA
print("\nBoston veri seti üzerinde EDA örnekleri:")
eda_preprocessor = DataPreprocessor(verbose=False)

# Eksik değer analizi
print("\n1. Eksik değer analizi:")
print(boston_data.isnull().sum())

# Aykırı değer analizi
print("\n2. Aykırı değer analizi:")
eda_preprocessor.original_data = boston_data
# Aykırı değerleri görselleştirmek için bu satırı aktif edin:
# eda_preprocessor.plot_outliers(columns=['CRIM', 'RM', 'LSTAT', 'PRICE'])

# Korelasyon matrisi
print("\n3. Korelasyon matrisi:")
corr = boston_data.corr()
print(corr['PRICE'].sort_values(ascending=False))
# Korelasyon matrisini görselleştirmek için bu satırı aktif edin:
# eda_preprocessor.plot_correlation_matrix()

# Özellik önemleri
print("\n4. Özellik önemleri:")
# Özellik önemlerini görselleştirmek için bu satırı aktif edin:
# eda_preprocessor.plot_feature_importance(target='PRICE')

print("\nÖrnek kullanım tamamlandı.")