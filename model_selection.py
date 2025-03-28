# -*- coding: utf-8 -*-
"""
Model Selection Module

Bu modül, farklı veri setleri için en iyi makine öğrenmesi algoritmasını seçer.
Regresyon, sınıflandırma ve kümeleme problemleri için kullanılabilir.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any

# Veri ön işleme için gerekli kütüphaneler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

# SMOTE için gerekli kütüphane
from imblearn.over_sampling import SMOTE

# Regresyon algoritmaları
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Sınıflandırma algoritmaları
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Kümeleme algoritmaları
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Optuna için gerekli kütüphane
import optuna
from optuna.samplers import TPESampler

# Uyarıları bastır
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Aykırı değerleri tespit edip kaldıran transformer sınıfı.
    """
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.mask_ = None
        
    def fit(self, X, y=None):
        if self.method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            
            self.mask_ = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
        
        elif self.method == 'zscore':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            z_scores = np.abs((X - mean) / std)
            self.mask_ = ~(z_scores > self.threshold).any(axis=1)
            
        return self
    
    def transform(self, X):
        if self.mask_ is None:
            return X
        return X[self.mask_]
    
    def get_feature_names_out(self, input_features=None):
        return input_features


class ModelSelectionModule:
    """
    Farklı veri setleri için en iyi makine öğrenmesi algoritmasını seçen modül.
    """
    def __init__(self, problem_type='classification', random_state=42, n_trials=100, n_jobs=-1, verbose=1):
        """
        Parameters
        ----------
        problem_type : str, default='classification'
            Problem tipi. 'classification', 'regression' veya 'clustering' olabilir.
        random_state : int, default=42
            Rastgele sayı üreteci için başlangıç değeri.
        n_trials : int, default=100
            Optuna optimizasyonu için deneme sayısı.
        n_jobs : int, default=-1
            Paralel işlem sayısı. -1, tüm işlemcileri kullanır.
        verbose : int, default=1
            Çıktı detay seviyesi.
        """
        self.problem_type = problem_type.lower()
        self.random_state = random_state
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.best_pipeline = None
        self.models_scores = {}
        self.feature_importances = None
        
        # Problem tipine göre modelleri ve değerlendirme metriklerini ayarla
        if self.problem_type == 'classification':
            self._setup_classification_models()
            self.scoring = 'f1_weighted'
            self.greater_is_better = True
        elif self.problem_type == 'regression':
            self._setup_regression_models()
            self.scoring = 'neg_mean_squared_error'
            self.greater_is_better = False
        elif self.problem_type == 'clustering':
            self._setup_clustering_models()
            self.scoring = 'silhouette'
            self.greater_is_better = True
        else:
            raise ValueError("Problem tipi 'classification', 'regression' veya 'clustering' olmalıdır.")
    
    def _setup_classification_models(self):
        """
        Sınıflandırma modellerini ayarlar.
        """
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=self.random_state),
            'KNN': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier(random_state=self.random_state),
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state),
            'SVC': SVC(probability=True, random_state=self.random_state),
            'NaiveBayes': GaussianNB(),
            'MLP': MLPClassifier(random_state=self.random_state),
            'XGBoost': XGBClassifier(random_state=self.random_state),
            'LightGBM': LGBMClassifier(random_state=self.random_state),
            'CatBoost': CatBoostClassifier(random_state=self.random_state, verbose=0)
        }
        
        # Ensemble modelleri için temel modeller
        self.ensemble_models = {
            'Bagging': BaggingClassifier(random_state=self.random_state),
            'Voting': VotingClassifier(estimators=[(name, model) for name, model in self.models.items()], voting='soft'),
            'Stacking': StackingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                final_estimator=LogisticRegression(random_state=self.random_state)
            )
        }
        
        # Tüm modelleri birleştir
        self.models.update(self.ensemble_models)
        
        # Değerlendirme metrikleri
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')
        }
    
    def _setup_regression_models(self):
        """
        Regresyon modellerini ayarlar.
        """
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state),
            'ElasticNet': ElasticNet(random_state=self.random_state),
            'KNN': KNeighborsRegressor(),
            'DecisionTree': DecisionTreeRegressor(random_state=self.random_state),
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state),
            'SVR': SVR(),
            'MLP': MLPRegressor(random_state=self.random_state),
            'XGBoost': XGBRegressor(random_state=self.random_state),
            'LightGBM': LGBMRegressor(random_state=self.random_state),
            'CatBoost': CatBoostRegressor(random_state=self.random_state, verbose=0)
        }
        
        # Ensemble modelleri için temel modeller
        self.ensemble_models = {
            'Bagging': BaggingRegressor(random_state=self.random_state),
            'Voting': VotingRegressor(estimators=[(name, model) for name, model in self.models.items()]),
            'Stacking': StackingRegressor(
                estimators=[(name, model) for name, model in self.models.items()],
                final_estimator=Ridge(random_state=self.random_state)
            )
        }
        
        # Tüm modelleri birleştir
        self.models.update(self.ensemble_models)
        
        # Değerlendirme metrikleri
        self.metrics = {
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'r2': r2_score
        }
    
    def _setup_clustering_models(self):
        """
        Kümeleme modellerini ayarlar.
        """
        self.models = {
            'KMeans': KMeans(random_state=self.random_state),
            'AgglomerativeClustering': AgglomerativeClustering(),
            'DBSCAN': DBSCAN()
        }
        
        # Değerlendirme metrikleri
        self.metrics = {
            'silhouette': silhouette_score,
            'calinski_harabasz': calinski_harabasz_score,
            'davies_bouldin': davies_bouldin_score
        }
    
    def _get_param_space(self, trial, model_name):
        """
        Model adına göre hiperparametre uzayını döndürür.
        
        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna denemesi.
        model_name : str
            Model adı.
            
        Returns
        -------
        dict
            Hiperparametre uzayı.
        """
        if self.problem_type == 'classification':
            return self._get_classification_param_space(trial, model_name)
        elif self.problem_type == 'regression':
            return self._get_regression_param_space(trial, model_name)
        elif self.problem_type == 'clustering':
            return self._get_clustering_param_space(trial, model_name)
    
    def _get_classification_param_space(self, trial, model_name):
        """
        Sınıflandırma modelleri için hiperparametre uzayını döndürür.
        """
        if model_name == 'LogisticRegression':
            return {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': self.random_state
            }
        elif model_name == 'KNN':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
        elif model_name == 'DecisionTree':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.random_state
            }
        elif model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.random_state
            }
        elif model_name == 'GradientBoosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'SVC':
            return {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'probability': True,
                'random_state': self.random_state
            }
        elif model_name == 'NaiveBayes':
            return {}
        elif model_name == 'MLP':
            return {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'random_state': self.random_state
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'LightGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'CatBoost':
            return {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'random_state': self.random_state,
                'verbose': 0
            }
        elif model_name == 'Bagging':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 5, 20),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'Voting' or model_name == 'Stacking':
            return {}
        else:
            return {}
    
    def _get_regression_param_space(self, trial, model_name):
        """
        Regresyon modelleri için hiperparametre uzayını döndürür.
        """
        if model_name == 'LinearRegression':
            return {}
        elif model_name == 'Ridge':
            return {
                'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
                'random_state': self.random_state
            }
        elif model_name == 'Lasso':
            return {
                'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
                'random_state': self.random_state
            }
        elif model_name == 'ElasticNet':
            return {
                'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
                'random_state': self.random_state
            }
        elif model_name == 'KNN':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
        elif model_name == 'DecisionTree':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.random_state
            }
        elif model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.random_state
            }
        elif model_name == 'GradientBoosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'SVR':
            return {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
        elif model_name == 'MLP':
            return {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'random_state': self.random_state
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'LightGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'CatBoost':
            return {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'random_state': self.random_state,
                'verbose': 0
            }
        elif model_name == 'Bagging':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 5, 20),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                'random_state': self.random_state
            }
        elif model_name == 'Voting' or model_name == 'Stacking':
            return {}
        else:
            return {}
    
    def _get_clustering_param_space(self, trial, model_name):
        """
        Kümeleme modelleri için hiperparametre uzayını döndürür.
        """
        if model_name == 'KMeans':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 10),
                'random_state': self.random_state
            }
        elif model_name == 'AgglomerativeClustering':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 10),
                'linkage': trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
            }
        elif model_name == 'DBSCAN':
            return {
                'eps': trial.suggest_float('eps', 0.1, 2.0),
                'min_samples': trial.suggest_int('min_samples', 2, 10)
            }
        else:
            return {}
    
    def _create_preprocessing_pipeline(self, X, categorical_features=None, numerical_features=None):
        """
        Veri ön işleme pipeline'ını oluşturur.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Özellik matrisi.
        categorical_features : list, default=None
            Kategorik özellik isimleri.
        numerical_features : list, default=None
            Sayısal özellik isimleri.
            
        Returns
        -------
        sklearn.pipeline.Pipeline
            Veri ön işleme pipeline'ı.
        """
        if categorical_features is None and numerical_features is None:
            # Otomatik özellik tipi tespiti
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('outlier_remover', OutlierRemover(method='iqr', threshold=1.5)),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline([
                    ('imputer', 'passthrough')
                ]), categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def _objective(self, trial, X, y, model_name, cv):
        """
        Optuna optimizasyonu için amaç fonksiyonu.
        
        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna denemesi.
        X : pandas.DataFrame
            Özellik matrisi.
        y : pandas.Series
            Hedef değişken.
        model_name : str
            Model adı.
        cv : sklearn.model_selection._split.BaseCrossValidator
            Çapraz doğrulama nesnesi.
            
        Returns
        -------
        float
            Değerlendirme metriği skoru.
        """
        # Hiperparametre uzayını al
        params = self._get_param_space(trial, model_name)
        
        # Modeli oluştur
        if model_name == 'Bagging':
            if self.problem_type == 'classification':
                base_estimator = DecisionTreeClassifier(random_state=self.random_state)
                model = BaggingClassifier(base_estimator=base_estimator, **params)
            else:  # regression
                base_estimator = DecisionTreeRegressor(random_state=self.random_state)
                model = BaggingRegressor(base_estimator=base_estimator, **params)
        elif model_name == 'Voting' or model_name == 'Stacking':
            # Ensemble modelleri için özel bir durum, bu modeller zaten kurulmuş durumda
            model = self.models[model_name]
        else:
            # Diğer modeller için
            model = clone(self.models[model_name])
            model.set_params(**params)
        
        # Çapraz doğrulama ile modeli değerlendir
        if self.problem_type == 'clustering':
            # Kümeleme için özel değerlendirme
            scores = []
            for n_clusters in range(2, 11):
                if model_name == 'KMeans' or model_name == 'AgglomerativeClustering':
                    model.set_params(n_clusters=n_clusters)
                try:
                    model.fit(X)
                    labels = model.labels_
                    score = silhouette_score(X, labels)
                    scores.append(score)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Hata: {e}")
                    scores.append(-np.inf)
            return np.max(scores) if scores else -np.inf
        else:
            # Sınıflandırma ve regresyon için çapraz doğrulama
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
                return np.mean(scores)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Hata: {e}")
                return -np.inf if not self.greater_is_better else np.inf
    
    def fit(self, X, y=None, categorical_features=None, numerical_features=None, 
            apply_smote=False, test_size=0.2, random_state=None):
        """
        Veri setini kullanarak en iyi modeli seçer ve eğitir.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Özellik matrisi.
        y : pandas.Series, default=None
            Hedef değişken. Kümeleme için None olabilir.
        categorical_features : list, default=None
            Kategorik özellik isimleri.
        numerical_features : list, default=None
            Sayısal özellik isimleri.
        apply_smote : bool, default=False
            SMOTE uygulanıp uygulanmayacağı. Sadece sınıflandırma için geçerlidir.
        test_size : float, default=0.2
            Test seti oranı.
        random_state : int, default=None
            Rastgele sayı üreteci için başlangıç değeri.
            
        Returns
        -------
        self : object
            Kendisi.
        """
        if random_state is None:
            random_state = self.random_state
        
        # Veri tiplerini kontrol et
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if y is not None and not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Veri ön işleme pipeline'ını oluştur
        self.preprocessor = self._create_preprocessing_pipeline(X, categorical_features, numerical_features)
        
        # Veri setini böl (kümeleme hariç)
        if self.problem_type != 'clustering':
            if y is None:
                raise ValueError("Sınıflandırma ve regresyon için y değeri gereklidir.")
            
            # Eğitim ve test setlerini ayır
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Veri ön işleme
            X_train_processed = self.preprocessor.fit_transform(X_train)
            
            # SMOTE uygula (sınıflandırma için)
            if apply_smote and self.problem_type == 'classification':
                try:
                    smote = SMOTE(random_state=random_state)
                    X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
                    if self.verbose > 0:
                        print("SMOTE uygulandı.")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"SMOTE uygulanamadı: {e}")
            
            # Çapraz doğrulama nesnesi
            if self.problem_type == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            else:  # regression
                cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
            
            # Her model için Optuna ile hiperparametre optimizasyonu
            self.models_scores = {}
            self.best_score = -np.inf if self.greater_is_better else np.inf
            self.best_model = None
            self.best_params = None
            
            for model_name in self.models.keys():
                if self.verbose > 0:
                    print(f"\nModel: {model_name} optimizasyonu başlıyor...")
                
                # Optuna çalışması
                study = optuna.create_study(
                    direction='maximize' if self.greater_is_better else 'minimize',
                    sampler=TPESampler(seed=random_state)
                )
                
                # Objective fonksiyonu
                objective = lambda trial: self._objective(
                    trial, X_train_processed, y_train, model_name, cv
                )
                
                # Optimizasyon
                try:
                    study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
                    
                    # En iyi skoru kaydet
                    best_score = study.best_value
                    best_params = study.best_params
                    
                    # Modeli güncelle
                    if model_name == 'Bagging':
                        if self.problem_type == 'classification':
                            base_estimator = DecisionTreeClassifier(random_state=random_state)
                            best_model = BaggingClassifier(base_estimator=base_estimator, **best_params)
                        else:  # regression
                            base_estimator = DecisionTreeRegressor(random_state=random_state)
                            best_model = BaggingRegressor(base_estimator=base_estimator, **best_params)
                    elif model_name == 'Voting' or model_name == 'Stacking':
                        best_model = self.models[model_name]
                    else:
                        best_model = clone(self.models[model_name])
                        best_model.set_params(**best_params)
                    
                    # Modeli eğit
                    best_model.fit(X_train_processed, y_train)
                    
                    # Test seti üzerinde değerlendir
                    X_test_processed = self.preprocessor.transform(X_test)
                    y_pred = best_model.predict(X_test_processed)
                    
                    # Metrikler
                    metrics_values = {}
                    for metric_name, metric_func in self.metrics.items():
                        try:
                            metrics_values[metric_name] = metric_func(y_test, y_pred)
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"Metrik hesaplanamadı {metric_name}: {e}")
                    
                    # Sonuçları kaydet
                    self.models_scores[model_name] = {
                        'best_score': best_score,
                        'best_params': best_params,
                        'test_metrics': metrics_values,
                        'model': best_model
                    }
                    
                    # En iyi modeli güncelle
                    if ((self.greater_is_better and best_score > self.best_score) or
                        (not self.greater_is_better and best_score < self.best_score)):
                        self.best_score = best_score
                        self.best_model = best_model
                        self.best_params = best_params
                    
                    if self.verbose > 0:
                        print(f"Model: {model_name}, En İyi Skor: {best_score:.4f}")
                        print(f"Test Metrikleri: {metrics_values}")
                
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Model {model_name} optimizasyonu başarısız: {e}")
            
            # Özellik önemini hesapla (mümkünse)
            self._calculate_feature_importance(X_train, X_train_processed)
            
            # En iyi modeli tüm veri seti üzerinde eğit
            X_processed = self.preprocessor.transform(X)
            if apply_smote and self.problem_type == 'classification':
                try:
                    smote = SMOTE(random_state=random_state)
                    X_processed, y = smote.fit_resample(X_processed, y)
                except Exception:
                    pass
            
            self.best_model.fit(X_processed, y)
            
            # En iyi pipeline'ı oluştur
            self.best_pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', self.best_model)
            ])
        
        else:  # Kümeleme için
            # Veri ön işleme
            X_processed = self.preprocessor.fit_transform(X)
            
            # Her model için Optuna ile hiperparametre optimizasyonu
            self.models_scores = {}
            self.best_score = -np.inf  # Kümeleme için daha yüksek silhouette skoru daha iyidir
            self.best_model = None
            self.best_params = None
            
            for model_name in self.models.keys():
                if self.verbose > 0:
                    print(f"\nModel: {model_name} optimizasyonu başlıyor...")
                
                # Optuna çalışması
                study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(seed=random_state)
                )
                
                # Objective fonksiyonu
                objective = lambda trial: self._objective(
                    trial, X_processed, None, model_name, None
                )
                
                # Optimizasyon
                try:
                    study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
                    
                    # En iyi skoru kaydet
                    best_score = study.best_value
                    best_params = study.best_params
                    
                    # Modeli güncelle
                    best_model = clone(self.models[model_name])
                    best_model.set_params(**best_params)
                    
                    # Modeli eğit
                    best_model.fit(X_processed)
                    
                    # Kümeleme metriklerini hesapla
                    try:
                        labels = best_model.labels_
                        metrics_values = {}
                        for metric_name, metric_func in self.metrics.items():
                            try:
                                metrics_values[metric_name] = metric_func(X_processed, labels)
                            except Exception as e:
                                if self.verbose > 0:
                                    print(f"Metrik hesaplanamadı {metric_name}: {e}")
                        
                        # Sonuçları kaydet
                        self.models_scores[model_name] = {
                            'best_score': best_score,
                            'best_params': best_params,
                            'metrics': metrics_values,
                            'model': best_model
                        }
                        
                        # En iyi modeli güncelle
                        if best_score > self.best_score:
                            self.best_score = best_score
                            self.best_model = best_model
                            self.best_params = best_params
                        
                        if self.verbose > 0:
                            print(f"Model: {model_name}, En İyi Skor: {best_score:.4f}")
                            print(f"Metrikler: {metrics_values}")
                    
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Model {model_name} değerlendirme başarısız: {e}")
                
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Model {model_name} optimizasyonu başarısız: {e}")
            
            # En iyi pipeline'ı oluştur
            self.best_pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', self.best_model)
            ])
        
        return self
    
    def _calculate_feature_importance(self, X, X_processed):
        """
        Özellik önemini hesaplar (mümkünse).
        
        Parameters
        ----------
        X : pandas.DataFrame
            Orijinal özellik matrisi.
        X_processed : numpy.ndarray
            İşlenmiş özellik matrisi.
        """
        try:
            # Özellik önemini destekleyen modeller için
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                self.feature_importances = pd.Series(
                    importances,
                    index=X.columns,
                    name='feature_importance'
                ).sort_values(ascending=False)
            elif hasattr(self.best_model, 'coef_'):
                # Lineer modeller için
                if len(self.best_model.coef_.shape) == 1:
                    # Regresyon veya ikili sınıflandırma
                    importances = np.abs(self.best_model.coef_)
                    self.feature_importances = pd.Series(
                        importances,
                        index=X.columns,
                        name='feature_importance'
                    ).sort_values(ascending=False)
                else:
                    # Çok sınıflı sınıflandırma
                    importances = np.mean(np.abs(self.best_model.coef_), axis=0)
                    self.feature_importances = pd.Series(
                        importances,
                        index=X.columns,
                        name='feature_importance'
                    ).sort_values(ascending=False)
        except Exception as e:
            if self.verbose > 0:
                print(f"Özellik önemi hesaplanamadı: {e}")
            self.feature_importances = None
    
    def predict(self, X):
        """
        Yeni veri için tahmin yapar.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Özellik matrisi.
            
        Returns
        -------
        numpy.ndarray
            Tahminler.
        """
        if self.best_pipeline is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        return self.best_pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Sınıflandırma için olasılık tahminleri yapar.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Özellik matrisi.
            
        Returns
        -------
        numpy.ndarray
            Olasılık tahminleri.
        """
        if self.problem_type != 'classification':
            raise ValueError("Bu metod sadece sınıflandırma için kullanılabilir.")
        
        if self.best_pipeline is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # predict_proba metodunu destekleyen modeller için
        if hasattr(self.best_model, 'predict_proba'):
            X_processed = self.preprocessor.transform(X)
            return self.best_model.predict_proba(X_processed)
        else:
            raise ValueError("Seçilen model olasılık tahminlerini desteklemiyor.")
    
    def get_best_model_summary(self):
        """
        En iyi model hakkında özet bilgi döndürür.
        
        Returns
        -------
        dict
            En iyi model özeti.
        """
        if self.best_model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # En iyi model adını bul
        best_model_name = None
        for model_name, model_info in self.models_scores.items():
            if model_info['model'] == self.best_model:
                best_model_name = model_name
                break
        
        summary = {
            'best_model': best_model_name,
            'best_score': self.best_score,
            'best_params': self.best_params
        }
        
        # Metrikler
        if best_model_name in self.models_scores:
            if self.problem_type == 'clustering':
                summary['metrics'] = self.models_scores[best_model_name].get('metrics', {})
            else:
                summary['test_metrics'] = self.models_scores[best_model_name].get('test_metrics', {})
        
        return summary
    
    def plot_feature_importance(self, top_n=10, figsize=(10, 6)):
        """
        Özellik önemini görselleştirir.
        
        Parameters
        ----------
        top_n : int, default=10
            Görselleştirilecek en önemli özellik sayısı.
        figsize : tuple, default=(10, 6)
            Şekil boyutu.
        """
        if self.feature_importances is None:
            print("Özellik önemi hesaplanamadı veya mevcut değil.")
            return
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importances.head(top_n)
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f"En Önemli {top_n} Özellik")
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, metric=None, figsize=(12, 6)):
        """
        Model karşılaştırmasını görselleştirir.
        
        Parameters
        ----------
        metric : str, default=None
            Karşılaştırılacak metrik. None ise, optimizasyon skoru kullanılır.
        figsize : tuple, default=(12, 6)
            Şekil boyutu.
        """
        if not self.models_scores:
            print("Model skorları mevcut değil.")
            return
        
        plt.figure(figsize=figsize)
        
        if metric is None:
            # Optimizasyon skorlarını kullan
            scores = {model_name: info['best_score'] for model_name, info in self.models_scores.items()}
            title = "Model Karşılaştırması - Optimizasyon Skoru"
        else:
            # Belirtilen metriği kullan
            scores = {}
            for model_name, info in self.models_scores.items():
                if self.problem_type == 'clustering':
                    if 'metrics' in info and metric in info['metrics']:
                        scores[model_name] = info['metrics'][metric]
                else:
                    if 'test_metrics' in info and metric in info['test_metrics']:
                        scores[model_name] = info['test_metrics'][metric]
            
            title = f"Model Karşılaştırması - {metric}"
        
        if not scores:
            print(f"Belirtilen metrik ({metric}) için skorlar mevcut değil.")
            return
        
        # Skorları sırala
        scores = pd.Series(scores).sort_values(ascending=not self.greater_is_better)
        
        # Görselleştir
        sns.barplot(x=scores.values, y=scores.index)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Modeli kaydeder.
        
        Parameters
        ----------
        filepath : str
            Kaydedilecek dosya yolu.
        """
        if self.best_pipeline is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Modeli kaydet
        joblib.dump(self.best_pipeline, filepath)
        
        # Model özetini de kaydet
        summary_filepath = os.path.splitext(filepath)[0] + '_summary.json'
        summary = self.get_best_model_summary()
        
        # JSON serileştirilebilir hale getir
        for key in summary:
            if key == 'best_params' and summary[key] is not None:
                # Serileştirilemeyecek nesneleri string'e dönüştür
                for param_key, param_value in summary[key].items():
                    if not isinstance(param_value, (int, float, str, bool, list, dict, type(None))):
                        summary[key][param_key] = str(param_value)
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"Model kaydedildi: {filepath}")
        print(f"Model özeti kaydedildi: {summary_filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Kaydedilmiş modeli yükler.
        
        Parameters
        ----------
        filepath : str
            Yüklenecek dosya yolu.
            
        Returns
        -------
        sklearn.pipeline.Pipeline
            Yüklenen model pipeline'ı.
        """
        return joblib.load(filepath)