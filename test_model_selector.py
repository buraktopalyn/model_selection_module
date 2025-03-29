# -*- coding: utf-8 -*-
"""
Model Seçici Test Dosyası

Bu dosya, model_selector.py dosyasındaki ModelSelector sınıfını test etmek için kullanılır.
"""

import sys
import numpy as np
from model_selector import ModelSelector

# Test verisi oluştur
np.random.seed(42)
X = np.random.rand(100, 5)
y_reg = np.random.rand(100) * 10  # Regresyon için hedef değişken
y_cls = np.random.randint(0, 3, 100)  # Sınıflandırma için hedef değişken (3 sınıf)

# Regresyon modellerini test et
print("\n=== Regresyon Modelleri ===")
ms_reg = ModelSelector('regression')
print("Yüklenen regresyon modelleri:")
for model_name in ms_reg.models.keys():
    print(f"- {model_name}")

# Sınıflandırma modellerini test et
print("\n=== Sınıflandırma Modelleri ===")
ms_cls = ModelSelector('classification')
print("Yüklenen sınıflandırma modelleri:")
for model_name in ms_cls.models.keys():
    print(f"- {model_name}")

print("\nTest başarılı bir şekilde tamamlandı!")