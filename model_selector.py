# -*- coding: utf-8 -*-
"""
Model Seçme Modülü

Bu modül, en popüler makine öğrenmesi algoritmalarını içerir ve en iyi modeli seçmeye yardımcı olur.
- 14 regresyon algoritması
- 14 sınıflandırma algoritması
- 5 kümeleme algoritması
- Ensemble modeller (bagging, stacking, boosting)
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Regresyon Modelleri
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Gradient Boosting Modelleri
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb

# Sınıflandırma Modelleri
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Kümeleme Modelleri
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch

# Ensemble Modeller için
from sklearn.ensemble import StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier

class ModelSelector:
    def __init__(self, problem_type: str = 'regression') -> None:
        """
        Model seçici sınıfını başlatır.

        Args:
            problem_type (str, optional): Problem tipi ('regression', 'classification', 'clustering').
                Varsayılan: 'regression'.

        Attributes:
            models (dict): Eğitilecek modellerin sözlüğü
            results (dict): Model sonuçlarını saklamak için sözlük
            best_model (Any): En iyi performans gösteren model
            best_model_name (Union[str, None]): En iyi modelin adı
            best_params (dict | None): En iyi model parametreleri
            best_score (float): En iyi skor değeri

        Raises:
            ValueError: Geçersiz problem tipi verildiğinde
        """
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = float('-inf') if problem_type != 'clustering' else float('inf')
        
        # Modelleri yükle
        self._load_models()
    
    def _load_models(self):
        """
        Problem tipine göre modelleri yükler.
        """
        if self.problem_type == 'regression':
            self.models = {
                'Linear Regression': LinearRegression(n_jobs=-1),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(n_jobs=-1),
                'AdaBoost': AdaBoostRegressor(),
                'Bagging': BaggingRegressor(n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_jobs=-1),
                'LightGBM': lgbm.LGBMRegressor(n_jobs=-1),
                'CatBoost': cb.CatBoostRegressor(verbose=0, thread_count=-1)
            }
        
        elif self.problem_type == 'classification':
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(),
                'SVC': SVC(probability=True),
                'KNN': KNeighborsClassifier(n_jobs=-1),
                'Naive Bayes': GaussianNB(),
                'LDA': LinearDiscriminantAnalysis(),
                'QDA': QuadraticDiscriminantAnalysis(),
                'AdaBoost': AdaBoostClassifier(),
                'Bagging': BaggingClassifier(n_jobs=-1),
                'XGBoost': xgb.XGBClassifier(n_jobs=-1),
                'LightGBM': lgbm.LGBMClassifier(n_jobs=-1),
                'CatBoost': cb.CatBoostClassifier(verbose=0, thread_count=-1)
            }
        
        elif self.problem_type == 'clustering':
            self.models = {
                'KMeans': KMeans(),
                'Agglomerative': AgglomerativeClustering(),
                'DBSCAN': DBSCAN(),
                'Spectral': SpectralClustering(),
                'Birch': Birch()
            }
    
    from typing import Union, List

    def add_ensemble_model(self, ensemble_type: str, base_models: Union[list, None] = None, weights: Union[List[float], None] = None) -> None:
        """
        Ensemble model ekler ve modeller sözlüğüne kaydeder.

        Args:
            ensemble_type (str): Ensemble tipi ('stacking', 'voting', 'bagging', 'boosting')
            base_models (list | None, optional): Temel modeller listesi. Varsayılan: İlk 3 model.
            weights (list[float] | None, optional): Voting modelleri için ağırlık listesi.

        Raises:
            ValueError: Geçersiz ensemble tipi veya model uyumsuzluğu durumunda
        """
        if base_models is None:
            # Use first 3 of existing models by default
            base_models = list(self.models.values())[:3]
        
        if self.problem_type == 'regression':
            if ensemble_type == 'stacking':
                self.models['Stacking'] = StackingRegressor(
                    estimators=[(f'model{i}', model) for i, model in enumerate(base_models)],
                    final_estimator=Ridge()
                )
            elif ensemble_type == 'voting':
                self.models['Voting'] = VotingRegressor(
                    estimators=[(f'model{i}', model) for i, model in enumerate(base_models)],
                    weights=weights
                )
            # Bagging and Boosting are already loaded
        
        elif self.problem_type == 'classification':
            if ensemble_type == 'stacking':
                self.models['Stacking'] = StackingClassifier(
                    estimators=[(f'model{i}', model) for i, model in enumerate(base_models)],
                    final_estimator=LogisticRegression()
                )
            elif ensemble_type == 'voting':
                self.models['Voting'] = VotingClassifier(
                    estimators=[(f'model{i}', model) for i, model in enumerate(base_models)],
                    weights=weights,
                    voting='soft'
                )
            # Bagging and Boosting are already loaded
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, None], test_size=0.2, random_state=42) -> None:
        """
        Tüm modelleri eğitir ve performanslarını değerlendirir.

        Args:
            X (pd.DataFrame): Eğitim veri seti özellikleri
            y (pd.Series | None): Hedef değişken serisi (kümeleme için None)
            test_size (float, optional): Test seti için ayrılacak oran. Varsayılan: 0.2.
            random_state (int, optional): Rastgelelik için seed değeri. Varsayılan: 42.

        Raises:
            ValueError: Hedef değişken eksik olduğunda veya problem tipiyle uyumsuzluk durumunda
        """
        # y is not used for clustering
        if self.problem_type != 'clustering':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            y_train = y_test = None
        
        # Başarılı model sayacı
        successful_models = 0
        
        for name, model in self.models.items():
            start_time = time.time()
            
            try:
                if self.problem_type != 'clustering':
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    if self.problem_type == 'regression':
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        self.results[name] = {
                            'model': model,
                            'r2': r2,
                            'rmse': rmse,
                            'train_time': train_time
                        }
                        
                        # Update best model (based on R2)
                        if r2 > self.best_score:
                            self.best_score = r2
                            self.best_model = model
                            self.best_model_name = name
                            self.best_params = model.get_params()
                        
                        successful_models += 1
                    
                    elif self.problem_type == 'classification':
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        
                        self.results[name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'classification_report': report,
                            'train_time': train_time
                        }
                        
                        # Update best model (based on accuracy)
                        if accuracy > self.best_score:
                            self.best_score = accuracy
                            self.best_model = model
                            self.best_model_name = name
                            self.best_params = model.get_params()
                        
                        successful_models += 1
                
                else:  # Kümeleme
                    # Bazı kümeleme algoritmaları fit_predict kullanır
                    if hasattr(model, 'fit_predict'):
                        model.fit_predict(X_train)
                    else:
                        model.fit(X_train)
                    
                    train_time = time.time() - start_time
                    
                    # Kümeleme için inertia veya silhouette score kullanılabilir
                    # Burada basitlik için sadece eğitim süresini kaydediyoruz
                    self.results[name] = {
                        'model': model,
                        'train_time': train_time
                    }
                    
                    # Kümeleme için en iyi model seçimi daha karmaşıktır
                    # Burada basitlik için en hızlı modeli seçiyoruz
                    if train_time < self.best_score:  # Kümeleme için en düşük süre
                        self.best_score = train_time
                        self.best_model = model
                        self.best_model_name = name
                        self.best_params = model.get_params()
                    
                    successful_models += 1
            
            except Exception as e:
                self.results[name] = {
                    'error': str(e)
                }
                print(f"Error training {name}: {str(e)}")
        
        # Eğer hiçbir model başarıyla eğitilmediyse, en azından bir modeli kaydet
        if successful_models == 0 and len(self.models) > 0:
            # İlk modeli al
            name = list(self.models.keys())[0]
            model = self.models[name]
            
            try:
                # Basit bir model oluştur (örneğin, sınıflandırma için DummyClassifier)
                if self.problem_type == 'classification':
                    from sklearn.dummy import DummyClassifier
                    model = DummyClassifier(strategy='most_frequent')
                    model.fit(X_train, y_train)
                    
                    # Sonuçları kaydet
                    self.best_model = model
                    self.best_model_name = "Fallback Dummy Classifier"
                    self.best_params = model.get_params()
                    self.best_score = 0.0  # Düşük bir skor
                    
                    self.results["Fallback Dummy Classifier"] = {
                        'model': model,
                        'accuracy': 0.0,
                        'train_time': 0.1
                    }
                    
                    print("Fallback model created due to training errors.")
            except Exception as e:
                print(f"Failed to create fallback model: {str(e)}")
        
        print("Eğitim işlemi bitti.")
        return self
    
    def get_best_model(self):
        """
        En iyi modeli ve parametrelerini döndürür.
        
        Returns:
        --------
        dict
            En iyi model bilgileri
        """
        return {
            'model_name': self.best_model_name,
            'model': self.best_model,
            'parameters': self.best_params,
            'score': self.best_score
        }
    
    def get_results(self):
        """
        Tüm modellerin sonuçlarını döndürür.
        
        Returns:
        --------
        dict
            Tüm modellerin sonuçları
        """
        return self.results
    
    def predict(self, X):
        """
        En iyi model ile tahmin yapar.
        
        Parameters:
        -----------
        X : array-like
            Özellikler
        
        Returns:
        --------
        array-like
            Tahminler
        """
        if self.best_model is None:
            raise ValueError("Önce modeli eğitmelisiniz.")
        
        if self.problem_type != 'clustering':
            return self.best_model.predict(X)
        else:
            # Kümeleme için fit_predict veya predict kullan
            if hasattr(self.best_model, 'predict'):
                return self.best_model.predict(X)
            else:
                return self.best_model.fit_predict(X)
                
    def explain_model(self, X, max_display=20, plot_type='bar', save_plot=False, filename=None):
        """
        SHAP (SHapley Additive exPlanations) değerlerini kullanarak en iyi model tahminlerini açıklar.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Açıklanacak veri seti (özellikler)
        max_display : int, optional
            Gösterilecek maksimum özellik sayısı. Varsayılan: 20.
        plot_type : str, optional
            Görselleştirme tipi ('bar', 'beeswarm', 'waterfall', 'force'). Varsayılan: 'bar'.
        save_plot : bool, optional
            Grafiği kaydetmek için True, ekranda göstermek için False. Varsayılan: False.
        filename : str, optional
            Kaydedilecek dosya adı. Belirtilmezse "shap_values_{model_name}.png" kullanılır.
            
        Returns:
        --------
        tuple
            (shap_values, explainer) - SHAP değerleri ve açıklayıcı nesnesi
            
        Raises:
        -------
        ValueError
            Desteklenmeyen model tipi durumunda
        """
        if self.problem_type == 'clustering':
            raise ValueError("SHAP explanations are not supported for clustering models.")
            
        # Sadece en iyi modeli kullan
        if self.best_model is None:
            # Eğer hiçbir model başarıyla eğitilmediyse, sonuçlardaki ilk modeli kullan
            if not self.results:
                raise ValueError("You must train the model first.")
            
            # Sonuçlardaki ilk başarılı modeli bul
            for name, result in self.results.items():
                if 'error' not in result and 'model' in result:
                    model = result['model']
                    model_name = name
                    break
            else:
                raise ValueError("No successfully trained model found.")
        else:
            model = self.best_model
            model_name = self.best_model_name
            
        # TreeExplainer için uygun modeller
        tree_models = ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost']
        
        try:
            # Model tipine göre uygun explainer'ı seç
            if any(tree_name in model_name for tree_name in tree_models):
                explainer = shap.TreeExplainer(model)
            else:
                # Diğer modeller için KernelExplainer kullan
                # Örnek veri seti oluştur (hesaplama süresini azaltmak için)
                if len(X) > 100:
                    background = shap.sample(X, 100)
                else:
                    background = X
                    
                # Tahmin fonksiyonunu belirle
                if self.problem_type == 'regression':
                    predict_fn = model.predict
                else:  # classification
                    # Olasılık değerlerini döndüren bir fonksiyon kullan
                    if hasattr(model, 'predict_proba'):
                        predict_fn = model.predict_proba
                    else:
                        predict_fn = model.predict
                        
                explainer = shap.KernelExplainer(predict_fn, background)
            
            # SHAP değerlerini hesapla
            shap_values = explainer.shap_values(X)
            
            # Görselleştirme
            plt.figure(figsize=(10, 8))
            if plot_type == 'bar':
                shap.summary_plot(shap_values, X, plot_type='bar', max_display=max_display, show=False)
            elif plot_type == 'beeswarm':
                shap.summary_plot(shap_values, X, max_display=max_display, show=False)
            elif plot_type == 'waterfall':
                # Waterfall plot için tek bir örnek gerekiyor
                # Güncel SHAP sürümlerinde waterfall_plot için Explanation nesnesi gerekiyor
                try:
                    if isinstance(shap_values, list):  # Sınıflandırma durumunda
                        base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        shap_explanation = shap.Explanation(values=shap_values[0][0], 
                                                         base_values=float(base_value),  # Tek bir skaler değer kullan
                                                         data=X.iloc[0].values,
                                                         feature_names=X.columns.tolist())
                    else:  # Regresyon durumunda
                        base_value = explainer.expected_value
                        shap_explanation = shap.Explanation(values=shap_values[0], 
                                                         base_values=float(base_value),  # Tek bir skaler değer kullan
                                                         data=X.iloc[0].values,
                                                         feature_names=X.columns.tolist())
                    
                    # max_display parametresini kullanarak waterfall plot oluştur
                    shap.plots.waterfall(shap_explanation, max_display=max_display)
                except Exception as e:
                    print(f"Waterfall plot oluşturulurken hata: {str(e)}")
                    # Alternatif olarak summary plot göster
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[0], X, plot_type='bar', max_display=max_display, show=False)
                    else:
                        shap.summary_plot(shap_values, X, plot_type='bar', max_display=max_display, show=False)
            elif plot_type == 'force':
                # Force plot için tek bir örnek gerekiyor
                try:
                    if len(X) > 1:
                        print("Force plot için ilk örnek kullanılıyor.")
                        X_single = X.iloc[0:1]
                        shap_values_single = explainer.shap_values(X_single)
                    else:
                        X_single = X
                        shap_values_single = shap_values
                    
                    # Sınıflandırma ve regresyon durumlarını ele al
                    if isinstance(shap_values_single, list):  # Sınıflandırma durumunda
                        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        shap.force_plot(expected_value, shap_values_single[0][0], X_single.iloc[0], show=False)
                    else:  # Regresyon durumunda
                        shap.force_plot(explainer.expected_value, shap_values_single[0], X_single.iloc[0], show=False)
                except Exception as e:
                    print(f"Force plot oluşturulurken hata: {str(e)}")
                    print("Alternatif olarak bar plot kullanılıyor...")
                    # Bar plot ile devam et
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[0], X, plot_type='bar', max_display=max_display, show=False)
                    else:
                        shap.summary_plot(shap_values, X, plot_type='bar', max_display=max_display, show=False)
            
            plt.title(f"{model_name} için SHAP Değerleri")
            plt.tight_layout()
            
            if save_plot:
                if filename is None:
                    filename = f"shap_values_{model_name.replace(' ', '_').lower()}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Grafik '{filename}' olarak kaydedildi.")
            else:
                plt.show()
                
            return shap_values, explainer
            
        except Exception as e:
            print(f"SHAP hesaplaması sırasında hata oluştu: {str(e)}")
            raise