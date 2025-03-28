# -*- coding: utf-8 -*-
"""
Model Selection Module - Örnek Kullanım

Bu dosya, Model Selection Module'ün nasıl kullanılacağını gösterir.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_boston, load_wine, make_blobs
from sklearn.model_selection import train_test_split
from model_selection import ModelSelectionModule

# Uyarıları bastır
import warnings
warnings.filterwarnings("ignore")


def classification_example():
    """
    Sınıflandırma örneği.
    """
    print("\n" + "=" * 50)
    print("Sınıflandırma Örneği")
    print("=" * 50)
    
    # Veri setini yükle
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    
    # Model seçim modülünü oluştur
    model_selector = ModelSelectionModule(
        problem_type='classification',
        random_state=42,
        n_trials=20,  # Örnek için az sayıda deneme
        verbose=1
    )
    
    # Modeli eğit
    model_selector.fit(X, y, apply_smote=True, test_size=0.2)
    
    # En iyi model özeti
    best_model_summary = model_selector.get_best_model_summary()
    print("\nEn İyi Model Özeti:")
    for key, value in best_model_summary.items():
        print(f"{key}: {value}")
    
    # Özellik önemini görselleştir
    print("\nÖzellik Önemi:")
    model_selector.plot_feature_importance()
    
    # Model karşılaştırmasını görselleştir
    print("\nModel Karşılaştırması:")
    model_selector.plot_model_comparison()
    
    # Modeli kaydet
    model_selector.save_model('classification_model.joblib')
    print("\nModel kaydedildi: classification_model.joblib")
    
    return model_selector


def regression_example():
    """
    Regresyon örneği.
    """
    print("\n" + "=" * 50)
    print("Regresyon Örneği")
    print("=" * 50)
    
    try:
        # Boston veri seti için alternatif (sklearn 1.0'dan sonra kaldırıldı)
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target, name='price')
    except:
        # Alternatif olarak yapay veri oluştur
        print("Boston veri seti yüklenemedi, yapay veri oluşturuluyor...")
        n_samples = 500
        n_features = 10
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(
            3 * X['feature_0'] + 2 * X['feature_1'] - X['feature_2'] + np.random.randn(n_samples),
            name='target'
        )
    
    # Model seçim modülünü oluştur
    model_selector = ModelSelectionModule(
        problem_type='regression',
        random_state=42,
        n_trials=20,  # Örnek için az sayıda deneme
        verbose=1
    )
    
    # Modeli eğit
    model_selector.fit(X, y, test_size=0.2)
    
    # En iyi model özeti
    best_model_summary = model_selector.get_best_model_summary()
    print("\nEn İyi Model Özeti:")
    for key, value in best_model_summary.items():
        print(f"{key}: {value}")
    
    # Özellik önemini görselleştir
    print("\nÖzellik Önemi:")
    model_selector.plot_feature_importance()
    
    # Model karşılaştırmasını görselleştir
    print("\nModel Karşılaştırması:")
    model_selector.plot_model_comparison()
    
    # Modeli kaydet
    model_selector.save_model('regression_model.joblib')
    print("\nModel kaydedildi: regression_model.joblib")
    
    return model_selector


def clustering_example():
    """
    Kümeleme örneği.
    """
    print("\n" + "=" * 50)
    print("Kümeleme Örneği")
    print("=" * 50)
    
    # Yapay veri oluştur
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    X = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
    
    # Model seçim modülünü oluştur
    model_selector = ModelSelectionModule(
        problem_type='clustering',
        random_state=42,
        n_trials=20,  # Örnek için az sayıda deneme
        verbose=1
    )
    
    # Modeli eğit
    model_selector.fit(X)
    
    # En iyi model özeti
    best_model_summary = model_selector.get_best_model_summary()
    print("\nEn İyi Model Özeti:")
    for key, value in best_model_summary.items():
        print(f"{key}: {value}")
    
    # Kümeleme sonuçlarını görselleştir
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Tahminleri al
    clusters = model_selector.predict(X)
    
    # Görselleştir
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='feature_0', y='feature_1', hue=clusters, data=X, palette='viridis')
    plt.title(f"Kümeleme Sonuçları - {best_model_summary['best_model']}")
    plt.tight_layout()
    plt.show()
    
    # Model karşılaştırmasını görselleştir
    print("\nModel Karşılaştırması:")
    model_selector.plot_model_comparison()
    
    # Modeli kaydet
    model_selector.save_model('clustering_model.joblib')
    print("\nModel kaydedildi: clustering_model.joblib")
    
    return model_selector


def main():
    """
    Ana fonksiyon.
    """
    print("Model Selection Module - Örnek Kullanım")
    
    # Sınıflandırma örneği
    classification_model = classification_example()
    
    # Regresyon örneği
    regression_model = regression_example()
    
    # Kümeleme örneği
    clustering_model = clustering_example()
    
    print("\nTüm örnekler tamamlandı.")


if __name__ == "__main__":
    main()