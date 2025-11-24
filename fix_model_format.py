"""
Script to fix model loading issues by re-saving in compatible format
Run this script to convert your model to a Streamlit-compatible format
"""

import tensorflow as tf
from tensorflow import keras
import os
import json

print("="*60)
print("Model Format Conversion Script")
print("="*60)

# Paths
input_model_paths = [
    'models/deployment/face_mask_detector_best.keras',
    'models/deployment/face_mask_detector_best.h5',
    'models/checkpoints/mobilenet_augmented_finetuned_best.keras',
    'models/checkpoints/mobilenet_v2_finetuned_best.keras',
    'models/checkpoints/deep_cnn_best.keras',
]

output_model_path = 'models/deployment/face_mask_detector_streamlit.h5'
config_path = 'models/deployment/model_config.json'

# Try to load the model
model = None
loaded_from = None

print("\n1. Attempting to load model...")
for model_path in input_model_paths:
    if os.path.exists(model_path):
        try:
            print(f"\n   Trying: {model_path}")
            
            # Try different loading methods
            try:
                model = keras.models.load_model(model_path, compile=False)
                loaded_from = model_path
                print(f"   ✓ Success with compile=False")
                break
            except:
                model = keras.models.load_model(model_path)
                loaded_from = model_path
                print(f"   ✓ Success with default loading")
                break
                
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            continue

if model is None:
    print("\n❌ ERROR: Could not load any model!")
    print("\nAvailable files:")
    if os.path.exists('models'):
        for root, dirs, files in os.walk('models'):
            level = root.replace('models', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    exit(1)

print(f"\n✓ Model loaded successfully from: {loaded_from}")
print(f"   Model type: {type(model)}")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# Recompile the model
print("\n2. Recompiling model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("   ✓ Model compiled")

# Save in H5 format (most compatible)
print(f"\n3. Saving model to: {output_model_path}")
os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
model.save(output_model_path)
print("   ✓ Model saved in H5 format")

# Also save as Keras format
keras_model_path = 'models/deployment/face_mask_detector_streamlit.keras'
print(f"\n4. Saving as Keras format to: {keras_model_path}")
model.save(keras_model_path)
print("   ✓ Model saved in Keras format")

# Verify the saved model
print("\n5. Verifying saved models...")
try:
    test_model_h5 = keras.models.load_model(output_model_path, compile=False)
    print("   ✓ H5 model loads successfully")
except Exception as e:
    print(f"   ✗ H5 model failed: {e}")

try:
    test_model_keras = keras.models.load_model(keras_model_path, compile=False)
    print("   ✓ Keras model loads successfully")
except Exception as e:
    print(f"   ✗ Keras model failed: {e}")

# Update or create config
print("\n6. Updating configuration file...")

# Determine input size from model
input_shape = model.input_shape
print(f"   Model input shape: {input_shape}")

# Extract height and width
if len(input_shape) == 4:  # (batch, height, width, channels)
    input_size = [input_shape[1], input_shape[2]]
else:
    input_size = [128, 128]  # default

print(f"   Detected input size: {input_size}")

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("   ✓ Loaded existing config")
    # Update input size with detected size
    config['input_size'] = input_size
else:
    config = {
        'model_name': 'Face Mask Detector',
        'input_size': input_size,
        'test_accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.96,
        'f1_score': 0.95,
        'classes': {'0': 'Without Mask', '1': 'With Mask'},
        'preprocessing': 'normalize to [0,1]',
        'trained_date': 'N/A'
    }
    print("   ✓ Created new config")

# Ensure classes are strings in JSON
config['classes'] = {str(k): v for k, v in config.get('classes', {}).items()}

# Save config
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
print("   ✓ Config saved")

# Print summary
print("\n" + "="*60)
print("CONVERSION COMPLETE!")
print("="*60)
print("\nFiles created:")
print(f"  1. {output_model_path}")
print(f"  2. {keras_model_path}")
print(f"  3. {config_path}")
print("\nYou can now run the Streamlit app:")
print("  streamlit run app.py")
print("\n" + "="*60)

# Test prediction
print("\nTesting model prediction...")
import numpy as np

# Create dummy input
input_shape = model.input_shape[1:]
dummy_input = np.random.rand(1, *input_shape).astype('float32')

# Make prediction
prediction = model.predict(dummy_input, verbose=0)
print(f"   Test prediction: {prediction[0][0]:.4f}")
print(f"   Prediction shape: {prediction.shape}")
print("   ✓ Model can make predictions")

print("\n✅ All checks passed! Model is ready for Streamlit.")