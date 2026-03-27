import random
import time

class LatencyProbe:

    @staticmethod
    def sample():
        base = random.randint(15,40)
        noise = random.uniform(-3,5)
        spike = 80 if random.random() < 0.05 else 0
        return int(base + noise + spike)

class FogNode:

    def __init__(self, name):
        self.name = name
        self.load = random.randint(25,55)
        self.temp = 42

    def evolve(self):
        drift = random.randint(-6,12)
        self.load = max(0, min(100, self.load + drift))
        self.temp = 30 + self.load * 0.7 + random.random()

    def snapshot(self):
        return {
            "id": self.name,
            "cpu": self.load,
            "temp": round(self.temp,1),
            "mode": "Adaptive"
        }

class Vehicle:

    def __init__(self, vid):
        self.vid = vid
        self.kind = random.choice(["Safety","Media"])
        self.payload = random.randint(20,400)

    def snapshot(self):
        return {"id": self.vid,"task":self.kind,"size_mb":self.payload}

class NetworkEnv:

    def __init__(self):
        self.nodes = [FogNode(f"Edge_{i}") for i in range(3)]
        self.vehicles = [Vehicle(f"Veh_{i}") for i in range(5)]
        self.latency = 0

    def step(self):
        for n in self.nodes:
            n.evolve()
        self.latency = LatencyProbe.sample()

    def state(self):
        return {
            "fog_nodes":[n.snapshot() for n in self.nodes],
            "vehicles":[v.snapshot() for v in self.vehicles],
            "real_latency_ms": self.latency
        }