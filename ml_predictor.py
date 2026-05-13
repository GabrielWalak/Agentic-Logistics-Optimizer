"""
ML Prediction Module — XGBoost delivery time estimation.
Trained on Brazilian E-Commerce (Olist) dataset.

If the model file is available, provides predicted_days from scenario features.
Falls back gracefully if model is not present.
"""
import os
import pickle
from typing import Optional, Dict, Any


MODEL_PATH = os.getenv("ML_MODEL_PATH", "./xgboost_model.pkl")

_model = None
_model_loaded = False


def _load_model():
    """Lazy-load the XGBoost model from disk."""
    global _model, _model_loaded
    if _model_loaded:
        return _model

    _model_loaded = True
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ ML model not found at {MODEL_PATH} — prediction disabled")
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print(f"✓ XGBoost model loaded from {MODEL_PATH}")
        return _model
    except Exception as e:
        print(f"⚠ Failed to load ML model: {e}")
        return None


def predict_delivery_days(
    distance_km: float,
    weight_g: float,
    freight_value: float,
    payment_lag_days: int = 0,
    is_weekend_order: int = 0,
    product_vol_cm3: float = 3000.0,
    customer_lat: float = -23.55,
    customer_lng: float = -46.63,
    seller_lat: float = -22.90,
    seller_lng: float = -43.17,
    purchase_month: int = 6,
) -> Optional[float]:
    """
    Predict delivery time using the XGBoost model.

    Args:
        distance_km: Distance between seller and customer
        weight_g: Product weight in grams
        freight_value: Shipping cost
        payment_lag_days: Days between order and payment
        is_weekend_order: 1 if ordered on weekend
        product_vol_cm3: Product volume (estimated if not provided)
        customer_lat/lng: Customer coordinates
        seller_lat/lng: Seller coordinates
        purchase_month: Month of purchase (1-12)

    Returns:
        Predicted delivery days, or None if model unavailable.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        # Feature order must match training pipeline
        features = [[
            weight_g,
            product_vol_cm3,
            distance_km,
            customer_lat,
            customer_lng,
            seller_lat,
            seller_lng,
            float(payment_lag_days),
            float(is_weekend_order),
            freight_value,
            float(purchase_month),
        ]]

        prediction = model.predict(features)
        predicted_days = float(prediction[0])

        # Clamp to reasonable range
        predicted_days = max(1.0, min(predicted_days, 60.0))
        return round(predicted_days, 1)

    except Exception as e:
        print(f"⚠ ML prediction failed: {e}")
        return None


def get_model_info() -> Dict[str, Any]:
    """Get model metadata for API responses."""
    model = _load_model()
    return {
        "model_available": model is not None,
        "model_path": MODEL_PATH,
        "model_type": "XGBoost Regressor",
        "dataset": "Brazilian E-Commerce (Olist)",
        "features": [
            "product_weight_g", "product_vol_cm3", "distance_km",
            "customer_lat", "customer_lng", "seller_lat", "seller_lng",
            "payment_lag_days", "is_weekend_order", "freight_value", "purchase_month"
        ],
        "known_limitations": [
            "No real-time weather data",
            "No carrier fleet availability",
            "No traffic/infrastructure disruptions",
            "Training data from 2016-2018"
        ],
    }


if __name__ == "__main__":
    info = get_model_info()
    print(f"Model available: {info['model_available']}")

    if info["model_available"]:
        # Test prediction
        result = predict_delivery_days(
            distance_km=800,
            weight_g=1200,
            freight_value=29.9,
            payment_lag_days=2,
            is_weekend_order=0,
        )
        print(f"Test prediction: {result} days")
