# -*- coding: utf-8 -*-
"""
Model Seçici Test Dosyası

Bu dosya, model_selector.py dosyasındaki ModelSelector sınıfının sınıflandırma özelliğini
test etmek için kullanılır. Scikit-learn'den iris veri seti kullanılarak en iyi
sınıflandırma modeli ve parametreleri bulunur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model_selector import ModelSelector

# Iris veri setini yükle
print("\n=== Iris Veri Seti Yükleniyor ===\n")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Veri seti hakkında bilgi ver
print(f"Veri seti boyutu: {X.shape}")
print(f"Özellikler: {feature_names}")
print(f"Sınıf sayısı: {len(target_names)}")
print(f"Sınıflar: {target_names}")

# Veri setini görselleştir
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=target_names[y], palette='viridis')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Iris Veri Seti - Sepal Özellikleri')

plt.subplot(2, 2, 2)
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=target_names[y], palette='viridis')
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title('Iris Veri Seti - Petal Özellikleri')

# Sınıflandırma modellerini test et
print("\n=== Sınıflandırma Modelleri Testi Başlıyor ===\n")
ms_cls = ModelSelector('classification')

# Modelleri eğit ve değerlendir
print("Modeller eğitiliyor ve değerlendiriliyor...")
ms_cls.fit(X, y, test_size=0.3, random_state=42)

# Sonuçları görüntüle
print("\n=== Model Sonuçları ===\n")
results = ms_cls.get_results()

# Sonuçları tablo halinde göster
accuracy_scores = {}
train_times = {}

for model_name, result in results.items():
    if 'error' in result:
        print(f"{model_name}: HATA - {result['error']}")
        continue
    
    accuracy = result['accuracy']
    train_time = result['train_time']
    
    accuracy_scores[model_name] = accuracy
    train_times[model_name] = train_time
    
    print(f"{model_name}:")
    print(f"  Doğruluk: {accuracy:.4f}")
    print(f"  Eğitim Süresi: {train_time:.4f} saniye")
    
    # Sınıflandırma raporu
    report = result['classification_report']
    print("  Sınıflandırma Raporu:")
    for class_name in target_names:
        class_idx = list(target_names).index(class_name)
        if str(class_idx) in report:
            class_report = report[str(class_idx)]
            print(f"    {class_name}:")
            print(f"      Precision: {class_report['precision']:.4f}")
            print(f"      Recall: {class_report['recall']:.4f}")
            print(f"      F1-score: {class_report['f1-score']:.4f}")
    print()

# En iyi modeli göster
best_model_info = ms_cls.get_best_model()
print("\n=== En İyi Model ===\n")
print(f"Model: {best_model_info['model_name']}")
print(f"Doğruluk Skoru: {best_model_info['score']:.4f}")
print("\nParametreler:")
for param, value in best_model_info['parameters'].items():
    print(f"  {param}: {value}")

# Modellerin doğruluk skorlarını görselleştir
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sorted_accuracy = {k: v for k, v in sorted(accuracy_scores.items(), key=lambda item: item[1], reverse=True)}
plt.bar(sorted_accuracy.keys(), sorted_accuracy.values(), color='skyblue')
plt.xticks(rotation=90)
plt.title('Model Doğruluk Skorları')
plt.ylabel('Doğruluk')
plt.ylim(0.8, 1.0)  # Doğruluk skorları genellikle yüksek olacaktır

# Modellerin eğitim sürelerini görselleştir
plt.subplot(1, 2, 2)
sorted_times = {k: v for k, v in sorted(train_times.items(), key=lambda item: item[1])}
plt.bar(sorted_times.keys(), sorted_times.values(), color='salmon')
plt.xticks(rotation=90)
plt.title('Model Eğitim Süreleri')
plt.ylabel('Süre (saniye)')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

print("\nTest başarılı bir şekilde tamamlandı!")
print("Model karşılaştırma grafiği 'model_comparison.png' olarak kaydedildi.")