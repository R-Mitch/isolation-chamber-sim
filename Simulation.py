
%%writefile simulation.py
import numpy as np
import matplotlib.pyplot as plt

def compute_shannon_entropy(data_bins):
    """Computes the exact Shannon entropy of discrete input states in bits."""
    value_counts = np.bincount(data_bins)
    probabilities = value_counts[value_counts > 0] / len(data_bins)
    return -np.sum(probabilities * np.log2(probabilities))

class IsolationChamberSimulation:
    def __init__(self, initial_energy=100.0, use_compression=True, compression_ratio=0.15):
        self.energy = initial_energy
        self.use_compression = use_compression
        self.compression_ratio = compression_ratio
        self.step_count = 0
        self.active = True
        
    def step(self, raw_input_signal):
        if not self.active:
            return None
        
        self.step_count += 1
        discretized = np.digitize(raw_input_signal, bins=np.linspace(-4, 4, 12))
        h_input = compute_shannon_entropy(discretized)
        
        if self.use_compression:
            effective_load = h_input * self.compression_ratio
            metabolic_cost = 0.15 + (effective_load * 0.35)
        else:
            effective_load = h_input
            metabolic_cost = 1.2 + (effective_load * 1.8)
            
        self.energy -= metabolic_cost
        if self.energy <= 0:
            self.active = False
            
        return {
            "step": self.step_count,
            "entropy": h_input,
            "energy": max(0.0, self.energy),
            "cost": metabolic_cost,
            "breach": not self.active
        }

if __name__ == "__main__":
    print("Isolation Chamber Simulation module loaded successfully.")
