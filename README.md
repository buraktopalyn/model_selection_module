# Model Seçme Modülü

Bu modül, makine öğrenmesi modellerini otomatik olarak değerlendiren ve en iyi performans gösteren modeli seçen bir araçtır. Regresyon, sınıflandırma ve kümeleme problemleri için çeşitli algoritmaları içerir. Ayrıca, veri ön işleme ve keşifsel veri analizi (EDA) için kapsamlı fonksiyonlar sunar.

## Özellikler

### Model Seçici (ModelSelector)

- **Çoklu Problem Tipi Desteği**: Regresyon, sınıflandırma ve kümeleme problemleri için kullanılabilir
- **Geniş Model Yelpazesi**: 
  - 14 regresyon algoritması
  - 14 sınıflandırma algoritması
  - 5 kümeleme algoritması
- **Ensemble Modeller**: Bagging, stacking, boosting ve voting yöntemleri
- **Otomatik Model Değerlendirme**: Tüm modelleri eğitir ve performanslarını karşılaştırır
- **En İyi Model Seçimi**: Performans metriklerine göre en iyi modeli otomatik olarak seçer

### Veri Ön İşleme (DataPreprocessor)

- **Eksik Değer İşleme**: Ortalama, medyan, mod, sabit değer, KNN ve satır silme yöntemleri
- **Aykırı Değer Tespiti ve İşleme**: Z-skor, IQR, izolasyon ormanı yöntemleri
- **Özellik Ölçeklendirme**: Standart, MinMax, Robust ve Power dönüşümleri
- **Kategorik Değişken Kodlama**: One-hot, Label ve Ordinal kodlama
- **Özellik Seçimi**: K-en iyi, önem tabanlı, RFE ve PCA yöntemleri
- **Keşifsel Veri Analizi (EDA)**: Görselleştirme araçları ve istatistiksel analizler

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Regresyon Örneği

```python
from model_selector import ModelSelector
import numpy as np

# Veri oluştur
X = np.random.rand(100, 5)
y = np.random.rand(100) * 10

# Model seçici oluştur
ms = ModelSelector(problem_type='regression')

# Modelleri eğit ve değerlendir
ms.fit(X, y)

# Sonuçları görüntüle
ms.display_results()

# En iyi modeli al
best_model = ms.get_best_model()
print(f"En iyi model: {ms.best_model_name}")
```

### Sınıflandırma Örneği

```python
from model_selector import ModelSelector
import numpy as np

# Veri oluştur
X = np.random.rand(100, 5)
y = np.random.randint(0, 3, 100)  # 3 sınıflı sınıflandırma

# Model seçici oluştur
ms = ModelSelector(problem_type='classification')

# Modelleri eğit ve değerlendir
ms.fit(X, y)

# Sonuçları görüntüle
ms.display_results()

# En iyi modeli al
best_model = ms.get_best_model()
print(f"En iyi model: {ms.best_model_name}")
```

### Kümeleme Örneği

```python
from model_selector import ModelSelector
import numpy as np

# Veri oluştur
X = np.random.rand(100, 5)

# Model seçici oluştur
ms = ModelSelector(problem_type='clustering')

# Modelleri eğit ve değerlendir
ms.fit(X, None)

# Sonuçları görüntüle
ms.display_results()
```

### Veri Ön İşleme Örneği

```python
from data_preprocessor import DataPreprocessor
import pandas as pd

# Veri oluştur
data = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [10, 20, 30, None, 50],
    'C': ['x', 'y', 'z', 'x', 'y']
})

# Ön işleme adımlarını tanımla
preprocessing_steps = {
    'handle_missing_values': {'method': 'mean'},
    'encode_categorical': {'method': 'one_hot'},
    'scale_features': {'method': 'standard'}
}

# Veri ön işleyici oluştur
preprocessor = DataPreprocessor(verbose=True)

# Veriyi işle
processed_data = preprocessor.fit_transform(
    data=data,
    preprocessing_steps=preprocessing_steps
)

print(processed_data.head())
```

### Ensemble Model Ekleme

```python
from model_selector import ModelSelector
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Veri oluştur
X = np.random.rand(100, 5)
y = np.random.rand(100) * 10

# Model seçici oluştur
ms = ModelSelector(problem_type='regression')

# Özel ensemble model ekle
base_models = [Ridge(), Lasso(), RandomForestRegressor()]
ms.add_ensemble_model('stacking', base_models=base_models)

# Modelleri eğit ve değerlendir
ms.fit(X, y)
```

### Tam Örnek Kullanım

Daha kapsamlı bir örnek için `example_usage.py` dosyasına bakabilirsiniz. Bu dosya, veri ön işleme ve model seçme modüllerinin birlikte nasıl kullanılacağını gösterir.

## Test Dosyaları

Proje, farklı problem tipleri için test dosyaları içerir:

- **test_regression.py**: Regresyon modelleri için test dosyası (California Housing veri seti)
- **test_classification.py**: Sınıflandırma modelleri için test dosyası (Iris veri seti)
- **test_clustering.py**: Kümeleme modelleri için test dosyası (yapay veri seti)

Test dosyalarını çalıştırmak için:

```bash
python test_regression.py
python test_classification.py
python test_clustering.py
```

## Desteklenen Modeller

### Regresyon Modelleri
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting
- SVR
- KNN
- AdaBoost
- Bagging
- XGBoost
- LightGBM
- CatBoost

### Sınıflandırma Modelleri
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVC
- KNN
- Naive Bayes
- LDA
- QDA
- AdaBoost
- Bagging
- XGBoost
- LightGBM
- CatBoost

### Kümeleme Modelleri
- KMeans
- Agglomerative
- DBSCAN
- Spectral
- Birch

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.