# 🚀 Farklı Senaryolar İçin Model Seçme Modülü

Bu modül, farklı veri setleri için en iyi makine öğrenmesi algoritmasını seçen profesyonel bir araçtır. Regresyon, sınıflandırma ve kümeleme problemleri için kullanılabilir.

## Özellikler

- **🔄 Çoklu Problem Tipi Desteği**: Regresyon, sınıflandırma ve kümeleme problemleri için kullanılabilir.
- **🔧 Otomatik Veri Ön İşleme**: 
  - Eksik verileri öğrenme yoluyla doldurma (KNN Imputer)
  - Aykırı değerleri tespit edip silme
  - Otomatik ölçeklendirme (Scaling)
  - Kategorik ve sayısal özellikleri otomatik tespit etme
- **⚖️ Dengesiz Veri Desteği**: Sınıflandırma problemleri için SMOTE uygulanabilir.
- **📚 Geniş Algoritma Desteği**:
  - **Regresyon**: Linear Regression, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, Gradient Boosting, SVR, MLP, XGBoost, LightGBM, CatBoost
  - **Sınıflandırma**: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVC, Naive Bayes, MLP, XGBoost, LightGBM, CatBoost
  - **Kümeleme**: KMeans, Agglomerative Clustering, DBSCAN
- **🤝 Ensemble Yöntemleri**: Bagging, Voting ve Stacking yöntemleri desteklenir.
- **🔍 Hiperparametre Optimizasyonu**: Optuna ile otomatik hiperparametre optimizasyonu.
- **📊 Kapsamlı Değerlendirme**: Çeşitli metriklerle model performansı değerlendirilir.
- **📈 Görselleştirme**: Özellik önemi ve model karşılaştırması için görselleştirme araçları.
- **💾 Model Kaydetme ve Yükleme**: Eğitilmiş modelleri kaydetme ve yükleme.

## Kullanım

1. 📥 Projeyi klonlayın:
   ```bash
   git clone https://github.com/kullanici/model_selection_module.git
   ```

2. 📦 Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. 🏃‍♂️ Örnek kullanım:
   ```python
   from model_selection import ModelSelector
   selector = ModelSelector(problem_type='classification')
   selector.fit(X_train, y_train)
   print(selector.best_model)
   ```

Bu adımlar, projeyi kurmanıza ve kullanmanıza yardımcı olacaktır. Herhangi bir sorunuz varsa, lütfen proje deposundaki sorunlar bölümüne bakın veya katkıda bulunun!