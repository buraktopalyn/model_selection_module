# -*- coding: utf-8 -*-
"""
Model Seçici Test Dosyası - Kümeleme

Bu dosya, model_selector.py dosyasındaki ModelSelector sınıfının kümeleme özelliğini
test etmek için kullanılır. Scikit-learn'den make_blobs fonksiyonu kullanılarak yapay
kümeleme veri seti oluşturulur ve en iyi kümeleme modeli bulunur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from model_selector import ModelSelector

# Yapay kümeleme veri seti oluştur
print("\n=== Yapay Kümeleme Veri Seti Oluşturuluyor ===\n")
n_samples = 500
n_features = 2
n_clusters = 4
random_state = 42

X, y = make_blobs(n_samples=n_samples, 
                 n_features=n_features, 
                 centers=n_clusters, 
                 cluster_std=1.0,
                 random_state=random_state)

# Veri seti hakkında bilgi ver
print(f"Veri seti boyutu: {X.shape}")
print(f"Özellik sayısı: {n_features}")
print(f"Gerçek küme sayısı: {n_clusters}")

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veri setini görselleştir
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis')
plt.title('Yapay Kümeleme Veri Seti - Gerçek Kümeler')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')

# Kümeleme modellerini test et
print("\n=== Kümeleme Modelleri Testi Başlıyor ===\n")
ms_clust = ModelSelector('clustering')

# Modelleri eğit ve değerlendir
print("Modeller eğitiliyor ve değerlendiriliyor...")
ms_clust.fit(X_scaled, None, test_size=0.3, random_state=42)

# Sonuçları görüntüle
print("\n=== Model Sonuçları ===\n")
results = ms_clust.get_results()

# Sonuçları tablo halinde göster
train_times = {}
silhouette_scores = {}

# Silhouette skorlarını hesapla
for name, result in results.items():
    if 'error' in result:
        print(f"{name}: HATA - {result['error']}")
        continue
    
    train_time = result['train_time']
    train_times[name] = train_time
    
    # Model nesnesini al
    model = result['model']
    
    # Modeli kullanarak kümeleme yap
    try:
        if hasattr(model, 'predict'):
            labels = model.predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
        
        # Silhouette skoru hesapla (en az 2 küme olmalı ve her kümede en az 1 örnek olmalı)
        n_labels = len(np.unique(labels))
        if n_labels > 1 and n_labels < len(X_scaled):
            score = silhouette_score(X_scaled, labels)
            silhouette_scores[name] = score
            print(f"{name}:")
            print(f"  Silhouette Skoru: {score:.4f}")
            print(f"  Eğitim Süresi: {train_time:.4f} saniye")
            print(f"  Bulunan Küme Sayısı: {n_labels}")
            print()
            
            # Her model için kümeleme sonuçlarını görselleştir
            plt.figure(figsize=(8, 6))
            plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.title(f'{name} Kümeleme Sonucu (Silhouette: {score:.4f})')
            plt.xlabel('Özellik 1')
            plt.ylabel('Özellik 2')
            plt.colorbar(label='Küme Etiketi')
            plt.savefig(f'clustering_{name}.png')
            plt.close()
        else:
            print(f"{name}: Silhouette skoru hesaplanamadı - Yetersiz küme sayısı")
    except Exception as e:
        print(f"{name}: Silhouette skoru hesaplanamadı - {str(e)}")

# En iyi modeli göster
best_model_info = ms_clust.get_best_model()
print("\n=== En İyi Model (Eğitim Süresine Göre) ===\n")
print(f"Model: {best_model_info['model_name']}")
print(f"Eğitim Süresi: {best_model_info['score']:.4f} saniye")
print("\nParametreler:")
for param, value in best_model_info['parameters'].items():
    print(f"  {param}: {value}")

# Silhouette skorlarına göre en iyi model
if silhouette_scores:
    best_silhouette_model = max(silhouette_scores.items(), key=lambda x: x[1])
    print("\n=== En İyi Model (Silhouette Skoruna Göre) ===\n")
    print(f"Model: {best_silhouette_model[0]}")
    print(f"Silhouette Skoru: {best_silhouette_model[1]:.4f}")

# Modellerin silhouette skorlarını görselleştir
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 2)
if silhouette_scores:
    sorted_scores = {k: v for k, v in sorted(silhouette_scores.items(), key=lambda item: item[1], reverse=True)}
    plt.bar(sorted_scores.keys(), sorted_scores.values(), color='skyblue')
    plt.xticks(rotation=90)
    plt.title('Model Silhouette Skorları')
    plt.ylabel('Silhouette Skoru')
    plt.ylim(0, 1.0)  # Silhouette skoru -1 ile 1 arasındadır, genelde pozitif değerler beklenir

# Modellerin eğitim sürelerini görselleştir
plt.subplot(2, 2, 3)
sorted_times = {k: v for k, v in sorted(train_times.items(), key=lambda item: item[1])}
plt.bar(sorted_times.keys(), sorted_times.values(), color='salmon')
plt.xticks(rotation=90)
plt.title('Model Eğitim Süreleri')
plt.ylabel('Süre (saniye)')

# En iyi modelin (silhouette skoruna göre) kümeleme sonucunu görselleştir
if silhouette_scores:
    plt.subplot(2, 2, 4)
    best_model_name = best_silhouette_model[0]
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'predict'):
        best_labels = best_model.predict(X_scaled)
    else:
        best_labels = best_model.fit_predict(X_scaled)
    
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_labels, cmap='viridis', alpha=0.7)
    plt.title(f'En İyi Model ({best_model_name}) Kümeleme Sonucu')
    plt.xlabel('Özellik 1')
    plt.ylabel('Özellik 2')
    plt.colorbar(label='Küme Etiketi')

plt.tight_layout()
plt.savefig('model_comparison_clustering.png')
plt.show()

print("\nTest başarılı bir şekilde tamamlandı!")
print("Model karşılaştırma grafiği 'model_comparison_clustering.png' olarak kaydedildi.")
print("Her bir modelin kümeleme sonuçları ayrı dosyalarda kaydedildi.")