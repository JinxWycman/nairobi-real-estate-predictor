#!/usr/bin/env python3
"""
Simple script to test model loading
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path

def test_model_load():
    """Test loading the trained model"""
    
    # Load feature info
    with open('models/model_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    print("✅ Feature info loaded successfully")
    print(f"Features: {len(feature_info['feature_names'])}")
    print(f"Model: {feature_info['model_name']}")
    print(f"R² Score: {feature_info['performance']['test_r2']:.4f}")
    
    # Try to load ensemble model
    model_path = Path('models/ensemble_best_model.pkl')
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("✅ Model loaded successfully")
    print(f"Model type: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print(f"Model keys: {list(model_data.keys())}")
        if 'meta_model' in model_data:
            print(f"Meta model type: {type(model_data['meta_model'])}")
        if 'base_models' in model_data:
            print(f"Base models: {len(model_data['base_models'])}")
    
    # Test prediction with sample data
    sample_features = np.zeros(len(feature_info['feature_names']))
    sample_features[feature_info['feature_names'].index('BEDROOMS')] = 3
    sample_features[feature_info['feature_names'].index('SIZE_SQM_CAPPED')] = 120
    sample_features[feature_info['feature_names'].index('LOCATION_Ruiru')] = 1
    sample_features[feature_info['feature_names'].index('IS_SATELLITE')] = 1
    
    # Make prediction
    if isinstance(model_data, dict) and 'meta_model' in model_data:
        prediction = model_data['meta_model'].predict([sample_features])[0]
    else:
        prediction = model_data.predict([sample_features])[0]
    
    print(f"✅ Sample prediction: KSh {prediction:,.0f}")
    
    return model_data, feature_info

if __name__ == "__main__":
    try:
        model, features = test_model_load()
        print("\n🎉 Model loading test PASSED!")
    except Exception as e:
        print(f"\n❌ Model loading test FAILED: {e}")
        import traceback
        traceback.print_exc()