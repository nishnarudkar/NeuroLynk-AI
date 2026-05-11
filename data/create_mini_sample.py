import json

# Path to the original sample
original_path = r'c:\Users\Asus\OneDrive\Desktop\NeuroLynk-AI\data\sample_patient_biomarkers.json'
output_path = r'c:\Users\Asus\OneDrive\Desktop\NeuroLynk-AI\data\mini_sample.json'

try:
    with open(original_path, 'r') as f:
        data = json.load(f)
    
    # Take only the first 50 features
    mini_features = data['features'][:50]
    
    # Pad with zeros to 753 so the ML model doesn't crash
    padded_features = mini_features + [0.0] * (753 - len(mini_features))
    
    payload = {
        "features": padded_features
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully created {output_path} with 50 real features (padded to 753).")

except Exception as e:
    print(f"Error: {e}")
