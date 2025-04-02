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

## Test Dosyaları 🧪

Proje, farklı problem tipleri için test dosyaları içerir:

- **test_regression.py**: Regresyon modelleri için test dosyası (California Housing veri seti)
- **test_classification.py**: Sınıflandırma modelleri için test dosyası (Iris veri seti)
- **test_clustering.py**: Kümeleme modelleri için test dosyası (sentetik veri seti)

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
