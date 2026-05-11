import pandas as pd
import json
import os

# Path to the CSV
csv_path = r'c:\Users\Asus\OneDrive\Desktop\NeuroLynk-AI\data\pd_speech_features.csv'
output_path = r'c:\Users\Asus\OneDrive\Desktop\NeuroLynk-AI\data\sample_patient_biomarkers.json'

try:
    # Load the CSV
    df = pd.read_csv(csv_path, header=1) # The original dataset has a double header
    
    # Take the first row of features (excluding ID and Class if they exist)
    # The dataset typically has ID as column 0 and Class as the last column.
    # We want the 753 features in between.
    
    # Check column count
    print(f"Total columns found: {len(df.columns)}")
    
    # We want exactly 753 features. 
    # Usually: Col 0 is ID, Cols 1-753 are features, Col 754 is Class.
    features = df.iloc[0, 1:754].tolist()
    
    if len(features) != 753:
        print(f"Warning: Extracted {len(features)} features, expected 753. Adjusting...")
        # Fallback: just take the first 753 numeric values found
        features = df.iloc[0].values[1:754].tolist()

    # Format for Prompt Opinion (wrapped in a "features" key which is standard)
    payload = {
        "features": [float(x) for x in features]
    }

    # Write to JSON
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully created {output_path} with {len(payload['features'])} features.")

except Exception as e:
    print(f"Error during conversion: {e}")
