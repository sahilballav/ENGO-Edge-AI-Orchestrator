import random
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class LatencyProbe:
    @staticmethod
    def sample():
        base = random.randint(15, 40)
        noise = random.uniform(-3, 5)
        spike = 80 if random.random() < 0.05 else 0
        return int(base + noise + spike)

class FogNode:
    def __init__(self, name):
        self.name = name
        self.load = 20  # Base idle load
        self.temp = 42.0

    # FIXED: Now accepts the camera proximity data!
    def evolve(self, camera_proximity=0):
        if camera_proximity > 0:
            # If an object gets close to the camera, spike the CPU compute load!
            self.load = int(min(100, 20 + camera_proximity))
        else:
            # Idle behavior when camera sees nothing
            drift = random.randint(-2, 2)
            self.load = max(10, min(40, self.load + drift))
            
        # Temperature dynamically reacts to the camera-driven CPU load
        self.temp = 30 + (self.load * 0.7) + (random.random() * 2)

class NetworkEnv:
    def __init__(self):
        self.nodes = [FogNode(f"Edge_{i}") for i in range(3)]
        self.vehicles = [Vehicle(f"Veh_{i}") for i in range(5)]
        self.latency = 0
        
        # Train ML Predictor on startup
        self._train_health_predictor()

    def _train_health_predictor(self):
        """Trains a Random Forest Classifier to predict Node Failure."""
        X_train = [
            [20, 45.0], [35, 50.2], [10, 38.5], [45, 55.0], 
            [65, 75.0], [70, 78.5], [60, 72.0], [75, 80.1], 
            [85, 90.0], [95, 95.5], [90, 92.0], [99, 98.0]  
        ]
        y_train = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        self.health_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.health_model.fit(X_train, y_train)

    def _predict_node_health(self, load, temp):
        """Uses the trained ML model to classify the live state of the node."""
        features = np.array([[load, temp]])
        prediction = self.health_model.predict(features)[0]
        
        if prediction == 0: 
            return "Stable"
        elif prediction == 1: 
            return "Warning"
        else: 
            return "CRITICAL"

    # FIXED: The step function now "catches" the camera data from app.py
    def step(self, live_camera_data=0):
        for n in self.nodes:
            n.evolve(live_camera_data)
            
        # Latency also spikes if an object is dangerously close to the camera!
        base_latency = 15 + (live_camera_data * 0.5)
        self.latency = int(base_latency + random.uniform(-2, 2))

    def state(self):
        return {
            "fog_nodes": [
                {
                    "id": n.name,
                    "cpu": n.load,
                    "temp": round(n.temp, 1),
                    "health_status": self._predict_node_health(n.load, n.temp),
                    "mode": "Adaptive"
                } for n in self.nodes
            ],
            "vehicles": [v.snapshot() for v in self.vehicles],
            "real_latency_ms": self.latency
        }

class Vehicle:
    def __init__(self, vid):
        self.vid = vid
        self.kind = random.choice(["Safety", "Media"])
        self.payload = random.randint(20, 400)

    def snapshot(self):
        return {"id": self.vid, "task": self.kind, "size_mb": self.payload}