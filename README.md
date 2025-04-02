# Model Selection Module 🚀

Bu modül, makine öğrenmesi modellerini otomatik olarak değerlendiren ve en iyi performans gösteren modeli seçen bir araçtır. Regresyon, sınıflandırma ve kümeleme problemleri için çeşitli algoritmalar içerir. Ayrıca, veri ön işleme ve keşifsel veri analizi (EDA) için kapsamlı fonksiyonlar sağlar.

## Özellikler 🌟

### Model Seçici (ModelSelector)

- **Çoklu Problem Tipi Desteği**: Regresyon, sınıflandırma ve kümeleme problemleri için kullanılabilir
- **Geniş Model Yelpazesi**: 
  - 14 regresyon algoritması
  - 14 sınıflandırma algoritması
  - 5 kümeleme algoritması
- **Ensemble Modeller**: Bagging, stacking, boosting ve voting yöntemleri
- **Otomatik Model Değerlendirme**: Tüm modelleri eğitir ve performanslarını karşılaştırır
- **En İyi Model Seçimi**: Performans metriklerine göre otomatik olarak en iyi modeli seçer

### Veri Ön İşleyici (DataPreprocessor)

- **Eksik Veri İşleme**: Çeşitli eksik veri doldurma yöntemleri
- **Aykırı Değer Tespiti ve İşleme**: Aykırı değerlerin tespiti ve düzeltilmesi
- **Özellik Ölçeklendirme**: Standartlaştırma, normalleştirme ve robust ölçeklendirme
- **Kategorik Değişken Kodlama**: Label encoding, one-hot encoding ve ordinal encoding
- **Özellik Seçimi**: Çeşitli özellik seçim yöntemleri
- **EDA Görselleştirmeleri**: Veri dağılımlarını ve ilişkilerini gösteren görselleştirmeler

## Kurulum ⚙️

1. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Modülü projenize ekleyin:
```python
from model_selector import ModelSelector
from data_preprocessor import DataPreprocessor
```

## Kullanım 📊

### Temel Kullanım
```python
# Model seçici oluştur
model_selector = ModelSelector(problem_type='regression')

# Veri ön işleyici oluştur
preprocessor = DataPreprocessor()

# Veriyi yükle ve ön işle
X, y = preprocessor.load_and_preprocess('data.csv')

# Modelleri eğit ve değerlendir
best_model = model_selector.select_best_model(X, y)
```

### Tam Örnek Kullanım
Daha kapsamlı bir örnek için `example_usage.py` dosyasına bakabilirsiniz. Bu dosya, veri ön işleme ve model seçme modüllerinin birlikte nasıl kullanılacağını gösterir.

```python
# Kendi modüllerimizi import et
from data_preprocessor import DataPreprocessor
from model_selector import ModelSelector
from sklearn.datasets import fetch_california_housing

# California ev fiyatları veri setini yükle
housing = fetch_california_housing()
california_data = pd.DataFrame(housing.data, columns=housing.feature_names)
california_data['PRICE'] = housing.target

# Data preprocessing
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
    data=california_data,
    target='PRICE',
    preprocessing_steps=preprocessing_steps
)

# Hedef değişkeni ayır
X = processed_data.drop(columns=['PRICE'])
y = processed_data['PRICE']

# Model seçimi
model_selector = ModelSelector(problem_type='regression')

# Ensemble modelleri ekle
model_selector.add_ensemble_model('stacking')
model_selector.add_ensemble_model('voting')

# Modelleri eğit
model_selector.fit(X, y)

# En iyi modeli al
best_model_info = model_selector.get_best_model()
print(f"En iyi regresyon modeli: {best_model_info['model_name']}")
print(f"R2 skoru: {best_model_info['score']:.4f}")
```

## Test Dosyaları 🧪

Proje, farklı problem tipleri için test dosyaları içerir:

### Regresyon Testi (test_regression.py)

California Housing veri seti kullanarak regresyon modellerini test eder.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from model_selector import ModelSelector

# California Housing veri setini yükle
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regresyon modellerini test et
ms_reg = ModelSelector('regression')

# Modelleri eğit ve değerlendir
ms_reg.fit(X_scaled, y, test_size=0.3, random_state=42)

# En iyi modeli göster
best_model_info = ms_reg.get_best_model()
print(f"Model: {best_model_info['model_name']}")
print(f"R² Skoru: {best_model_info['score']:.4f}")
print("\nParametreler:")
for param, value in best_model_info['parameters'].items():
    print(f"  {param}: {value}")
```

### Sınıflandırma Testi (test_classification.py)

Iris veri seti kullanarak sınıflandırma modellerini test eder.

```python
from sklearn.datasets import load_iris
from model_selector import ModelSelector

# Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Sınıflandırma modellerini test et
ms_cls = ModelSelector('classification')

# Modelleri eğit ve değerlendir
ms_cls.fit(X, y, test_size=0.3, random_state=42)

# En iyi modeli göster
best_model_info = ms_cls.get_best_model()
print(f"Model: {best_model_info['model_name']}")
print(f"Doğruluk Skoru: {best_model_info['score']:.4f}")
print("\nParametreler:")
for param, value in best_model_info['parameters'].items():
    print(f"  {param}: {value}")
```

### Kümeleme Testi (test_clustering.py)

Yapay veri seti oluşturarak kümeleme modellerini test eder.

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from model_selector import ModelSelector

# Yapay kümeleme veri seti oluştur
X, y = make_blobs(n_samples=500, 
                 n_features=2, 
                 centers=4, 
                 cluster_std=1.0,
                 random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kümeleme modellerini test et
ms_clust = ModelSelector('clustering')

# Modelleri eğit ve değerlendir
ms_clust.fit(X_scaled, None, test_size=0.3, random_state=42)

# Sonuçları göster
results = ms_clust.get_results()

# Silhouette skorlarını hesapla ve göster
for name, result in results.items():
    if 'error' in result:
        continue
    
    model = result['model']
    
    # Kümeleme yap
    if hasattr(model, 'predict'):
        labels = model.predict(X_scaled)
    else:
        labels = model.fit_predict(X_scaled)
    
    # Silhouette skorunu hesapla
    n_labels = len(np.unique(labels))
    if n_labels > 1 and n_labels < len(X_scaled):
        score = silhouette_score(X_scaled, labels)
        print(f"{name}: Silhouette Skoru: {score:.4f}")
```

Testleri çalıştırmak için:
```bash
python test_regression.py
python test_classification.py
python test_clustering.py
```

## Proje Yapısı 📂
```
model_selection_module/
├── data_preprocessor.py    # Veri ön işleme modülü
├── model_selector.py       # Model seçme modülü
├── example_usage.py        # Örnek kullanım
├── test_regression.py      # Regresyon testleri
├── test_classification.py  # Sınıflandırma testleri
├── test_clustering.py      # Kümeleme testleri
├── requirements.txt         # Gereksinimler
└── README.md               # Bu dosya
```

## Katkıda Bulunma 🤝

Katkıda bulunmak isterseniz:
1. Bu repo'yu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/awesome-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some awesome feature'`)
4. Branch'inizi pushlayın (`git push origin feature/awesome-feature`)
5. Bir Pull Request açın

## Lisans 📜
Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.
