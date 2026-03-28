import random
import time
import numpy as np
import pandas as pd
import os
import csv
import joblib

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
        self.load = 20  
        self.temp = 42.0

    def evolve(self, camera_proximity=0):
        if camera_proximity > 0:
            self.load = int(min(100, 20 + camera_proximity))
        else:
            drift = random.randint(-2, 2)
            self.load = max(10, min(40, self.load + drift))
            
        self.temp = 30 + (self.load * 0.7) + (random.random() * 2)

class NetworkEnv:
    def __init__(self):
        self.nodes = [FogNode(f"Edge_{i}") for i in range(3)]
        self.vehicles = [Vehicle(f"Veh_{i}") for i in range(5)]
        self.latency = 0
        
        self.model_path = 'models/rf_health_model.pkl'
        self.telemetry_file = 'telemetry_log.csv'
        self.last_model_time = 0 
        
        self._initialize_telemetry_database()
        self._check_for_new_brain()

    def _initialize_telemetry_database(self):
        # The Seed Data to prevent the "Cold Start" problem
        if not os.path.exists(self.telemetry_file):
            print("Creating new Telemetry Database...")
            with open(self.telemetry_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cpu_load', 'temperature', 'status'])
                writer.writerows([
                    [20, 45.0, 0], [35, 50.2, 0],
                    [65, 75.0, 1], [70, 78.5, 1],
                    [95, 95.5, 2], [99, 98.0, 2]
                ])

    def _check_for_new_brain(self):
        """Safely checks and loads the model, handling race conditions."""
        if os.path.exists(self.model_path):
            try:
                current_time = os.path.getmtime(self.model_path)
                # If the file is newer AND has actual data in it (not 0 bytes)
                if current_time > self.last_model_time and os.path.getsize(self.model_path) > 0:
                    self.health_model = joblib.load(self.model_path)
                    self.last_model_time = current_time
                    # print("🧠 Brain updated successfully!")
            except (EOFError, Exception):
                # If the file is being written to, just skip this update and try again next frame
                pass 
        else:
            self.health_model = None

    def _predict_node_health(self, load, temp):
        self._check_for_new_brain()
        
        if not hasattr(self, 'health_model') or self.health_model is None:
            return "UNKNOWN"
            
        features = np.array([[load, temp]])
        prediction = self.health_model.predict(features)[0]
        
        if prediction == 0: return "Stable"
        elif prediction == 1: return "Warning"
        else: return "CRITICAL"

    def step(self, live_camera_data=0):
        for n in self.nodes:
            n.evolve(live_camera_data)
            
            # Write live physical reality straight to the database!
            if live_camera_data > 0:
                if n.load > 85 or n.temp > 85.0: label = 2      
                elif n.load > 60 or n.temp > 65.0: label = 1    
                else: label = 0                                 
                
                with open(self.telemetry_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([n.load, n.temp, label])
                
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