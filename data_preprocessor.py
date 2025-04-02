# -*- coding: utf-8 -*-
"""
Data Preprocessing Module

This module contains various functions for data preprocessing and exploratory data analysis (EDA).
Features:
- Missing value handling
- Outlier detection and handling
- Feature scaling
- Categorical variable encoding
- Feature selection
- Exploratory data analysis (EDA) visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from schemas import PreprocessingConfig

# Görselleştirme ayarları
sns.set(style="whitegrid")
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, verbose: bool = True, config: Union[PreprocessingConfig, dict] = None) -> None:
        """
        Initializes the data preprocessing class.

        Args:
            verbose (bool, optional): Whether to show process details. Default: True.

        Attributes:
            transformers (dict): Dictionary to store scalers
            encoders (dict): Dictionary to store encoders
            imputers (dict): Dictionary to store missing value imputers
            feature_selectors (dict): Dictionary to store feature selectors
            outlier_indices (dict): Dictionary to store outlier indices
            dropped_columns (list): List of dropped columns
            original_data (pd.DataFrame): Original dataset
            processed_data (pd.DataFrame): Processed dataset
        """
        self.verbose = verbose
        if isinstance(config, dict):
            config = PreprocessingConfig(**config)
        elif config is None:
            config = PreprocessingConfig()
            
        self.config = config
        self.transformers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.outlier_indices = {}
        self.dropped_columns = []
        self.original_data = None
        self.processed_data = None
        
    def fit_transform(self, data: pd.DataFrame, target: Union[str, None] = None, preprocessing_steps: Union[dict, None] = None) -> pd.DataFrame:
        """
        Processes and transforms the data.

        Args:
            data (pd.DataFrame): Raw dataset to be processed
            target (str | None, optional): Target variable name. Default: None.
            preprocessing_steps (dict | None, optional): Dictionary containing preprocessing steps to apply.
                Format: {'step_name': {parameters}}. Default: None.

        Returns:
            pd.DataFrame: Processed dataset

        Raises:
            ValueError: When an invalid preprocessing step is provided
        """
        self.original_data = data.copy()
        self.processed_data = data.copy()
        
        # Eğer preprocessing_steps parametresi verilmişse, config yerine onu kullan
        if preprocessing_steps is not None:
            steps = preprocessing_steps
            # Eksik adımları varsayılan değerlerle doldur
            if 'drop_columns' not in steps:
                steps['drop_columns'] = {'columns': []}
        else:
            # Pydantic modelinden ön işleme adımlarını al
            steps = {
                'drop_columns': {'columns': self.config.drop_columns},
                'handle_missing_values': self.config.handle_missing_values,
                'handle_outliers': self.config.handle_outliers,
                'encode_categorical': self.config.encode_categorical,
                'scale_features': self.config.scale_features,
                'feature_selection': self.config.feature_selection
            }
        
        # Sütunları düşür
        if steps['drop_columns']['columns']:
            self._drop_columns(steps['drop_columns']['columns'])
        
        # Eksik değerleri işle
        if 'handle_missing_values' in steps:
            self._handle_missing_values(**steps['handle_missing_values'])
        
        # Aykırı değerleri işle
        if 'handle_outliers' in steps:
            self._handle_outliers(**steps['handle_outliers'])
        
        # Kategorik değişkenleri kodla
        if 'encode_categorical' in steps:
            self._encode_categorical(**steps['encode_categorical'])
        
        # Özellikleri ölçeklendir
        if 'scale_features' in steps:
            self._scale_features(**steps['scale_features'])
        
        # Özellik seçimi yap
        if 'feature_selection' in steps and target is not None:
            self._feature_selection(target=target, **steps['feature_selection'])
        
        if self.verbose:
            print("Veri ön işleme tamamlandı.")
            print(f"Orijinal veri boyutu: {self.original_data.shape}")
            print(f"İşlenmiş veri boyutu: {self.processed_data.shape}")
        
        return self.processed_data
    
    def _drop_columns(self, columns):
        """
        Drops specified columns.
        
        Parameters:
        -----------
        columns : list
            Names of columns to drop
        """
        if not columns:
            return
        
        # Var olan sütunları kontrol et
        existing_columns = [col for col in columns if col in self.processed_data.columns]
        
        if existing_columns:
            self.processed_data = self.processed_data.drop(columns=existing_columns)
            self.dropped_columns.extend(existing_columns)
            
            if self.verbose:
                print(f"Düşürülen sütunlar: {existing_columns}")
    
    def _handle_missing_values(self, method='mean', columns=None, **kwargs):
        """
        Handles missing values.
        
        Parameters:
        -----------
        method : str, default='mean'
            Missing value imputation method: 'mean', 'median', 'mode', 'constant', 'knn', 'drop'
        columns : list, optional
            Columns to process. If None, all columns are processed.
        **kwargs : dict
            Additional parameters
        """
        if columns is None:
            # Sayısal ve kategorik sütunları ayır
            numeric_columns = self.processed_data.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = self.processed_data.select_dtypes(exclude=['number']).columns.tolist()
        else:
            # Belirtilen sütunları veri tipine göre ayır
            numeric_columns = [col for col in columns if col in self.processed_data.select_dtypes(include=['number']).columns]
            categorical_columns = [col for col in columns if col in self.processed_data.select_dtypes(exclude=['number']).columns]
        
        # Eksik değer sayısını göster
        if self.verbose:
            missing_count = self.processed_data.isnull().sum()
            missing_count = missing_count[missing_count > 0]
            if not missing_count.empty:
                print("Eksik değer sayıları:")
                print(missing_count)
            else:
                print("Veri setinde eksik değer bulunmuyor.")
                return
        
        # Eksik değerleri doldur
        if method == 'drop':
            # Satırları düşür
            self.processed_data = self.processed_data.dropna(subset=columns if columns else None)
            if self.verbose:
                print(f"Eksik değer içeren satırlar düşürüldü. Yeni boyut: {self.processed_data.shape}")
        else:
            # Sayısal sütunlar için
            if numeric_columns:
                if method == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif method == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif method == 'constant':
                    fill_value = kwargs.get('fill_value', 0)
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                elif method == 'knn':
                    n_neighbors = kwargs.get('n_neighbors', 5)
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                else:
                    raise ValueError(f"Geçersiz sayısal eksik değer doldurma yöntemi: {method}")
                
                # Sayısal sütunları doldur
                self.processed_data[numeric_columns] = imputer.fit_transform(self.processed_data[numeric_columns])
                self.imputers['numeric'] = imputer
            
            # Kategorik sütunlar için
            if categorical_columns:
                if method == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                elif method == 'constant':
                    fill_value = kwargs.get('fill_value', 'unknown')
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                
                # Kategorik sütunları doldur
                self.processed_data[categorical_columns] = imputer.fit_transform(self.processed_data[categorical_columns])
                self.imputers['categorical'] = imputer
            
            if self.verbose:
                print(f"Eksik değerler '{method}' yöntemi ile dolduruldu.")
    
    def _handle_outliers(self, method='none', columns=None, threshold=1.5):
        """
        Handles outliers.
        
        Parameters:
        -----------
        method : str, default='none'
            Outlier handling method: 'none', 'remove', 'clip', 'winsorize', 'zscore'
        columns : list, optional
            Columns to process. If None, all numeric columns are processed.
        threshold : float, default=1.5
            Outlier threshold (for IQR method)
        """
        if method == 'none':
            return
        
        if columns is None:
            # Tüm sayısal sütunları seç
            columns = self.processed_data.select_dtypes(include=['number']).columns.tolist()
        else:
            # Belirtilen sütunlardan sayısal olanları seç
            columns = [col for col in columns if col in self.processed_data.select_dtypes(include=['number']).columns]
        
        outlier_indices = {}
        
        for col in columns:
            if method in ['remove', 'clip', 'winsorize']:
                # IQR yöntemi
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Aykırı değerleri bul
                outliers = self.processed_data[(self.processed_data[col] < lower_bound) | 
                                              (self.processed_data[col] > upper_bound)].index
                outlier_indices[col] = outliers
                
                if method == 'clip':
                    # Aykırı değerleri sınırla
                    self.processed_data[col] = self.processed_data[col].clip(lower_bound, upper_bound)
                    if self.verbose and len(outliers) > 0:
                        print(f"{col} sütununda {len(outliers)} aykırı değer sınırlandırıldı.")
                
                elif method == 'winsorize':
                    # Winsorize yöntemi
                    self.processed_data[col] = stats.mstats.winsorize(self.processed_data[col], 
                                                                     limits=[0.05, 0.05])
                    if self.verbose and len(outliers) > 0:
                        print(f"{col} sütununda {len(outliers)} aykırı değer winsorize edildi.")
            
            elif method == 'zscore':
                # Z-score yöntemi
                z_scores = np.abs(stats.zscore(self.processed_data[col], nan_policy='omit'))
                outliers = self.processed_data[z_scores > 3].index
                outlier_indices[col] = outliers
        
        # Aykırı değerleri kaldır (tüm sütunlardaki aykırı değerleri birleştir)
        if method == 'remove':
            all_outliers = set()
            for indices in outlier_indices.values():
                all_outliers.update(indices)
            
            if all_outliers:
                self.processed_data = self.processed_data.drop(index=all_outliers)
                if self.verbose:
                    print(f"Toplam {len(all_outliers)} aykırı değer içeren satır kaldırıldı.")
        
        self.outlier_indices = outlier_indices
    
    def _encode_categorical(self, method='one_hot', columns=None, **kwargs):
        """
        Encodes categorical variables.
        
        Parameters:
        -----------
        method : str, default='one_hot'
            Encoding method: 'one_hot', 'label', 'ordinal'
        columns : list, optional
            Columns to process. If None, all categorical columns are processed.
        **kwargs : dict
            Additional parameters
        """
        if columns is None:
            # Tüm kategorik sütunları seç
            columns = self.processed_data.select_dtypes(exclude=['number']).columns.tolist()
        else:
            # Belirtilen sütunlardan kategorik olanları seç
            columns = [col for col in columns if col in self.processed_data.columns and 
                      col not in self.processed_data.select_dtypes(include=['number']).columns]
        
        if not columns:
            if self.verbose:
                print("Kodlanacak kategorik sütun bulunamadı.")
            return
        
        if method == 'one_hot':
            # One-Hot Encoding
            drop_first = kwargs.get('drop_first', True)
            encoder = OneHotEncoder(sparse=False, drop='first' if drop_first else None)
            
            # Kategorik sütunları dönüştür
            encoded_data = encoder.fit_transform(self.processed_data[columns])
            
            # Yeni sütun isimlerini oluştur
            feature_names = encoder.get_feature_names_out(columns)
            
            # Kodlanmış verileri DataFrame'e dönüştür
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=self.processed_data.index)
            
            # Orijinal kategorik sütunları kaldır ve kodlanmış sütunları ekle
            self.processed_data = self.processed_data.drop(columns=columns).join(encoded_df)
            
            # Encoder'ı kaydet
            self.encoders['one_hot'] = encoder
            
            if self.verbose:
                print(f"{len(columns)} kategorik sütun one-hot encoding ile kodlandı.")
                print(f"Eklenen yeni sütunlar: {len(feature_names)}")
        
        elif method == 'label':
            # Label Encoding
            for col in columns:
                encoder = LabelEncoder()
                self.processed_data[col] = encoder.fit_transform(self.processed_data[col].astype(str))
                self.encoders[f'label_{col}'] = encoder
            
            if self.verbose:
                print(f"{len(columns)} kategorik sütun label encoding ile kodlandı.")
        
        elif method == 'ordinal':
            # Ordinal Encoding
            categories = kwargs.get('categories', None)
            encoder = OrdinalEncoder(categories=categories)
            
            self.processed_data[columns] = encoder.fit_transform(self.processed_data[columns])
            self.encoders['ordinal'] = encoder
            
            if self.verbose:
                print(f"{len(columns)} kategorik sütun ordinal encoding ile kodlandı.")
        
        else:
            raise ValueError(f"Geçersiz kategorik kodlama yöntemi: {method}")
    
    def _scale_features(self, method='standard', columns=None):
        """
        Scales numerical features.
        
        Parameters:
        -----------
        method : str, default='standard'
            Scaling method: 'standard', 'minmax', 'robust', 'power'
        columns : list, optional
            Columns to process. If None, all numeric columns are processed.
        """
        if columns is None:
            # Tüm sayısal sütunları seç
            columns = self.processed_data.select_dtypes(include=['number']).columns.tolist()
        else:
            # Belirtilen sütunlardan sayısal olanları seç
            columns = [col for col in columns if col in self.processed_data.select_dtypes(include=['number']).columns]
        
        if not columns:
            if self.verbose:
                print("Ölçeklendirilecek sayısal sütun bulunamadı.")
            return
        
        if method == 'standard':
            # Standart Ölçeklendirme (z-score)
            scaler = StandardScaler()
        elif method == 'minmax':
            # Min-Max Ölçeklendirme
            scaler = MinMaxScaler()
        elif method == 'robust':
            # Robust Ölçeklendirme
            scaler = RobustScaler()
        elif method == 'power':
            # Power Transformer (Yeo-Johnson)
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError(f"Geçersiz ölçeklendirme yöntemi: {method}")
        
        # Sütunları ölçeklendir
        self.processed_data[columns] = scaler.fit_transform(self.processed_data[columns])
        
        # Scaler'ı kaydet
        self.transformers[method] = scaler
        
        if self.verbose:
            print(f"{len(columns)} sayısal sütun '{method}' yöntemi ile ölçeklendirildi.")
    
    def _feature_selection(self, target, method='none', k=10, **kwargs):
        """
        Performs feature selection.
        
        Parameters:
        -----------
        target : str
            Target variable name
        method : str, default='none'
            Feature selection method: 'none', 'kbest', 'rfe', 'pca', 'importance'
        k : int, default=10
            Number of features to select
        **kwargs : dict
            Additional parameters
        """
        if method == 'none':
            return
        
        # Hedef değişkeni ayır
        if target in self.processed_data.columns:
            X = self.processed_data.drop(columns=[target])
            y = self.processed_data[target]
        else:
            if self.verbose:
                print(f"Hedef değişken '{target}' veri setinde bulunamadı.")
            return
        
        # Sayısal sütunları seç
        numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_columns:
            if self.verbose:
                print("Özellik seçimi için sayısal sütun bulunamadı.")
            return
        
        # Sadece sayısal sütunları kullan
        X_numeric = X[numeric_columns]
        
        if method == 'kbest':
            # K-Best özellik seçimi
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                # Sınıflandırma problemi
                score_func = kwargs.get('score_func', f_classif)
                if score_func == 'mutual_info':
                    score_func = mutual_info_classif
                selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_columns)))
            else:
                # Regresyon problemi
                score_func = kwargs.get('score_func', f_regression)
                if score_func == 'mutual_info':
                    score_func = mutual_info_regression
                selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_columns)))
            
            # Özellikleri seç
            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            # Seçilen özellikleri kaydet
            self.feature_selectors['kbest'] = selector
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                # Sınıflandırma problemi
                estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=100, random_state=42))
            else:
                # Regresyon problemi
                estimator = kwargs.get('estimator', RandomForestRegressor(n_estimators=100, random_state=42))
            
            selector = RFE(estimator=estimator, n_features_to_select=min(k, len(numeric_columns)), step=1)
            
            # Özellikleri seç
            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            # Seçilen özellikleri kaydet
            self.feature_selectors['rfe'] = selector
            
        elif method == 'pca':
            # Principal Component Analysis
            n_components = min(k, len(numeric_columns))
            selector = PCA(n_components=n_components)
            
            # Özellikleri dönüştür
            X_selected = selector.fit_transform(X_numeric)
            
            # Yeni sütun isimleri oluştur
            selected_features = [f'PC{i+1}' for i in range(n_components)]
            
            # PCA bileşenlerini DataFrame'e dönüştür
            X_pca = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            # Orijinal sayısal sütunları kaldır ve PCA bileşenlerini ekle
            non_numeric_columns = X.columns.difference(numeric_columns).tolist()
            self.processed_data = pd.concat([X[non_numeric_columns], X_pca, pd.DataFrame(y, columns=[target])], axis=1)
            
            # Seçilen özellikleri kaydet
            self.feature_selectors['pca'] = selector
            
            if self.verbose:
                print(f"PCA ile {n_components} bileşen oluşturuldu.")
                print(f"Açıklanan varyans oranı: {np.sum(selector.explained_variance_ratio_):.2f}")
            
            # PCA için erken dön
            return
            
        elif method == 'importance':
            # Özellik önemine göre seçim
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                # Sınıflandırma problemi
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                # Regresyon problemi
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Modeli eğit
            model.fit(X_numeric, y)
            
            # Özellik önemlerini al
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # En önemli k özelliği seç
            selected_indices = indices[:min(k, len(numeric_columns))]
            selected_features = X_numeric.columns[selected_indices].tolist()
            
            # Seçilen özellikleri kaydet
            self.feature_selectors['importance'] = model
            
            if self.verbose:
                print("Özellik önemleri:")
                for i, idx in enumerate(indices[:min(k, len(numeric_columns))]):
                    print(f"{i+1}. {X_numeric.columns[idx]}: {importances[idx]:.4f}")
        
        # PCA dışındaki yöntemler için
        if method != 'pca':
            # Seçilmeyen sütunları düşür
            columns_to_drop = [col for col in numeric_columns if col not in selected_features]
            if columns_to_drop:
                self.processed_data = self.processed_data.drop(columns=columns_to_drop)
                self.dropped_columns.extend(columns_to_drop)
            
            if self.verbose:
                print(f"{method} yöntemi ile {len(selected_features)} özellik seçildi.")
                print(f"Seçilen özellikler: {selected_features}")
    
    def plot_missing_values(self):
        """
        Visualizes missing values.
        """
        if self.original_data is None:
            print("Önce veriyi yüklemelisiniz.")
            return
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.original_data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Eksik Değerler')
        plt.tight_layout()
        plt.show()
        
        # Eksik değer yüzdeleri
        missing = (self.original_data.isnull().sum() / len(self.original_data)) * 100
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if not missing.empty:
            plt.figure(figsize=(10, 6))
            missing.plot(kind='bar')
            plt.title('Eksik Değer Yüzdeleri')
            plt.ylabel('Eksik Değer Yüzdesi')
            plt.xlabel('Sütunlar')
            plt.tight_layout()
            plt.show()
        else:
            print("Veri setinde eksik değer bulunmuyor.")
    
    def plot_outliers(self, columns=None, method='boxplot'):
        """
        Visualizes outliers.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to visualize. If None, all numeric columns are visualized.
        method : str, default='boxplot'
            Visualization method: 'boxplot', 'histogram'
        """
        if self.original_data is None:
            print("Önce veriyi yüklemelisiniz.")
            return
        
        if columns is None:
            # Tüm sayısal sütunları seç
            columns = self.original_data.select_dtypes(include=['number']).columns.tolist()
        else:
            # Belirtilen sütunlardan sayısal olanları seç
            columns = [col for col in columns if col in self.original_data.select_dtypes(include=['number']).columns]
        
        if not columns:
            print("Görselleştirilecek sayısal sütun bulunamadı.")
            return
        
        if method == 'boxplot':
            # Box plot
            plt.figure(figsize=(12, len(columns) * 2))
            for i, col in enumerate(columns, 1):
                plt.subplot(len(columns), 1, i)
                sns.boxplot(x=self.original_data[col])
                plt.title(f'{col} - Box Plot')
            plt.tight_layout()
            plt.show