import onnxruntime as ort
import numpy as np
import os
import json
from datetime import datetime

class DelayPredictor:
    def __init__(self, model_path: str):
        if model_path.endswith('.pkl'):
            model_path = model_path.replace('.pkl', '.onnx')

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

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
        input_data = np.array([[float(data[f]) for f in self.features]], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: input_data})

        predicted_delay = float(outputs[0][0][0])

        severity = "NORMAL"
        if predicted_delay > 30:
            severity = "CRITICAL"
        elif predicted_delay > 15:
            severity = "SIGNIFICANT"

        return {
            "predicted_delay_minutes": round(predicted_delay, 2),
            "severity": severity,
            "impact_description": f"Expected delay of {round(predicted_delay, 1)} minutes due to operational conditions.",
            "mitigation_strategies": [
                {
                    "strategy": "Dynamic Platform Re-assignment",
                    "implementation": "Re-route inbound trains to less congested platforms.",
                    "expected_reduction": "5-10 minutes"
                }
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def simulate_scenario(self, baseline: dict, modifications: dict):
        scenario = baseline.copy()
        scenario.update(modifications)

        baseline_res = self.predict(baseline)
        modified_res = self.predict(scenario)

        reduction = baseline_res['predicted_delay_minutes'] - modified_res['predicted_delay_minutes']
        improvement_pct = (reduction / baseline_res['predicted_delay_minutes'] * 100) if baseline_res['predicted_delay_minutes'] > 0 else 0

        return {
            "baseline": baseline_res,
            "modified_scenario": modified_res,
            "delay_reduction_minutes": round(reduction, 2),
            "improvement_percentage": round(improvement_pct, 1),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
