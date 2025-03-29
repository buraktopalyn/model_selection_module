# Model Seçme Modülü

Bu modül, makine öğrenmesi modellerini otomatik olarak değerlendiren ve en iyi performans gösteren modeli seçen bir araçtır. Regresyon, sınıflandırma ve kümeleme problemleri için çeşitli algoritmaları içerir.

## Özellikler

- **Çoklu Problem Tipi Desteği**: Regresyon, sınıflandırma ve kümeleme problemleri için kullanılabilir
- **Geniş Model Yelpazesi**: 
  - 14 regresyon algoritması
  - 14 sınıflandırma algoritması
  - 5 kümeleme algoritması
- **Ensemble Modeller**: Bagging, stacking, boosting ve voting yöntemleri
- **Otomatik Model Değerlendirme**: Tüm modelleri eğitir ve performanslarını karşılaştırır
- **En İyi Model Seçimi**: Performans metriklerine göre en iyi modeli otomatik olarak seçer

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