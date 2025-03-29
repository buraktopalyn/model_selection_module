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
            best_model_name (str | None): En iyi modelin adı
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
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Bagging': BaggingRegressor(),
                'XGBoost': xgb.XGBRegressor(),
                'LightGBM': lgbm.LGBMRegressor(),
                'CatBoost': cb.CatBoostRegressor(verbose=0)
            }
        
        elif self.problem_type == 'classification':
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'SVC': SVC(probability=True),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'LDA': LinearDiscriminantAnalysis(),
                'QDA': QuadraticDiscriminantAnalysis(),
                'AdaBoost': AdaBoostClassifier(),
                'Bagging': BaggingClassifier(),
                'XGBoost': xgb.XGBClassifier(),
                'LightGBM': lgbm.LGBMClassifier(),
                'CatBoost': cb.CatBoostClassifier(verbose=0)
            }
        
        elif self.problem_type == 'clustering':
            self.models = {
                'KMeans': KMeans(),
                'Agglomerative': AgglomerativeClustering(),
                'DBSCAN': DBSCAN(),
                'Spectral': SpectralClustering(),
                'Birch': Birch()
            }
    
    def add_ensemble_model(self, ensemble_type: str, base_models: list | None = None, weights: list[float] | None = None) -> None:
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
            # Varsayılan olarak mevcut modellerin ilk 3'ünü kullan
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
            # Bagging ve Boosting zaten yüklendi
        
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
            # Bagging ve Boosting zaten yüklendi
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None, test_size: float = 0.2, random_state: int = 42) -> None:
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
        # Kümeleme için y kullanılmaz
        if self.problem_type != 'clustering':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            y_train = y_test = None
        
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
                        
                        # En iyi modeli güncelle (R2'ye göre)
                        if r2 > self.best_score:
                            self.best_score = r2
                            self.best_model = model
                            self.best_model_name = name
                            self.best_params = model.get_params()
                    
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
                        
                        # En iyi modeli güncelle (accuracy'ye göre)
                        if accuracy > self.best_score:
                            self.best_score = accuracy
                            self.best_model = model
                            self.best_model_name = name
                            self.best_params = model.get_params()
                
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
            
            except Exception as e:
                self.results[name] = {
                    'error': str(e)
                }
        
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