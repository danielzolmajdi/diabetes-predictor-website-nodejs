import sys
import json
import torch
import joblib
import numpy as np
from trainingModel import DiabetesModel  # Import your model class

# Load the trained model and scaler
model = DiabetesModel(input_size=21)  # 21 features
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

scaler = joblib.load('full_feature_scaler.pkl')

# Read input from Node.js
input_json = sys.argv[1]
input_data = json.loads(input_json)

# Scale input using the saved scaler
scaled = scaler.transform([input_data])

# Convert to tensor and predict
input_tensor = torch.tensor(scaled, dtype=torch.float32)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

print(prediction)
