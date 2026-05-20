import onnxruntime as ort
import numpy as np
import os
import json
from datetime import datetime

class ConflictDetector:
    def __init__(self, model_path: str):
        if model_path.endswith('.pkl'):
            # Fallback/Legacy if needed, but we prefer .onnx
            model_path = model_path.replace('.pkl', '.onnx')

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Load features list from metadata if available
        meta_path = os.path.join(os.path.dirname(model_path), "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.features = json.load(f)['features']
        else:
            self.features = [
                'trains_in_section', 'available_platforms', 'platform_utilization',
                'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
                'is_peak_hour'
            ]

    def predict(self, data: dict):
        # Prepare input array
        input_data = np.array([[float(data[f]) for f in self.features]], dtype=np.float32)

        # Inference
        # RandomForestClassifier returns [label, probabilities]
        outputs = self.session.run(None, {self.input_name: input_data})

        # label = outputs[0][0]
        # probabilities = outputs[1][0] (dict {0: p0, 1: p1} in some opsets, or array)

        # Usually for RandomForest in ONNX:
        # outputs[0] is class labels
        # outputs[1] is list of dictionaries with probabilities
        prob_dict = outputs[1][0]
        conflict_prob = float(prob_dict[1])

        risk_level = "LOW"
        if conflict_prob > 0.7:
            risk_level = "HIGH"
        elif conflict_prob > 0.4:
            risk_level = "MEDIUM"

        recommendations = []
        if risk_level == "HIGH":
            recommendations.append({
                "priority": "CRITICAL",
                "action": "Increase headways",
                "details": "High conflict risk detected. Recommend increasing minimum spacing between trains."
            })

        return {
            "conflict_probability": round(conflict_prob, 4),
            "risk_level": risk_level,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
