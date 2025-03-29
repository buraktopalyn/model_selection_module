from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

class PreprocessingConfig(BaseModel):
    drop_columns: List[str] = Field(default_factory=list, description="Düşürülecek sütunlar")
    handle_missing_values: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "mean"},
        description="Eksik değer doldurma yöntemi ve parametreleri"
    )
    handle_outliers: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "none"},
        description="Aykırı değer işleme yöntemi ve parametreleri"
    )
    encode_categorical: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "one_hot", "columns": []},
        description="Kategorik değişken kodlama yöntemi ve parametreleri"
    )
    scale_features: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "standard", "columns": []},
        description="Özellik ölçeklendirme yöntemi ve parametreleri"
    )
    feature_selection: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "none", "k": 10},
        description="Özellik seçimi yöntemi ve parametreleri"
    )

    @validator("handle_missing_values")
    def validate_missing_values_method(cls, v):
        valid_methods = ["mean", "median", "mode", "constant", "knn"]
        if v["method"] not in valid_methods:
            raise ValueError(f"Geçersiz eksik değer doldurma yöntemi. Geçerli yöntemler: {valid_methods}")
        return v

    @validator("handle_outliers")
    def validate_outliers_method(cls, v):
        valid_methods = ["none", "zscore", "iqr", "isolation_forest"]
        if v["method"] not in valid_methods:
            raise ValueError(f"Geçersiz aykırı değer işleme yöntemi. Geçerli yöntemler: {valid_methods}")
        return v

class ModelSelectorConfig(BaseModel):
    problem_type: Literal["regression", "classification", "clustering"] = Field(
        default="regression",
        description="Problem tipi"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Test seti için ayrılacak oran"
    )
    random_state: int = Field(
        default=42,
        description="Rastgelelik için seed değeri"
    )

class EnsembleConfig(BaseModel):
    ensemble_type: Literal["stacking", "voting", "bagging", "boosting"] = Field(
        description="Ensemble model tipi"
    )
    base_models: Optional[List[str]] = Field(
        default=None,
        description="Temel model isimleri listesi"
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Voting modelleri için ağırlık listesi"
    )

    @validator("weights")
    def validate_weights(cls, v, values):
        if v is not None:
            if values.get("ensemble_type") != "voting":
                raise ValueError("Ağırlıklar sadece voting ensemble için kullanılabilir")
            if not all(w > 0 for w in v):
                raise ValueError("Tüm ağırlıklar pozitif olmalıdır")
            if abs(sum(v) - 1.0) > 1e-10:
                raise ValueError("Ağırlıkların toplamı 1 olmalıdır")
        return v