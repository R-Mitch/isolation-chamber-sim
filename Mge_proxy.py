import math
import numpy as np

# --- Core Logic Classes ---
class EgoLoopEngine:
    @staticmethod
    def calculate_shannon_entropy(tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        freqs = {}
        for token in tokens:
            freqs[token] = freqs.get(token, 0) + 1
        
        entropy = 0.0
        total = len(tokens)
        for count in freqs.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def compute_relevance_scores(tokens: list[str], task_keywords: list[str]) -> np.ndarray:
        scores = np.ones(len(tokens)) * 0.1
        for i, token in enumerate(tokens):
            if any(kw.lower() in token.lower() for kw in task_keywords):
                scores[i] = 1.0
        return scores


class MetabolicGatingProxy:
    def __init__(self, baseline_cost: float = 0.05):
        self.baseline_cost = baseline_cost

    def process_request(self, prompt: str, energy_budget: float, task_context: str) -> dict:
        tokens = prompt.split()
        task_keywords = task_context.split() if task_context else []
        
        raw_entropy = EgoLoopEngine.calculate_shannon_entropy(tokens)
        relevance = EgoLoopEngine.compute_relevance_scores(tokens, task_keywords)
        
        budget_ratio = max(0.01, min(1.0, energy_budget / 100.0))
        tau = (1.0 / budget_ratio) * 0.2
        
        pruned_tokens = []
        for token, score in zip(tokens, relevance):
            if score >= (tau * 0.5) or budget_ratio > 0.8:
                pruned_tokens.append(token)
                
        compressed_prompt = " ".join(pruned_tokens)
        burn_rate = self.baseline_cost + (0.01 * len(pruned_tokens)) + (0.05 * raw_entropy * (1.0 - budget_ratio))
        updated_budget = max(0.0, energy_budget - burn_rate)
        
        return {
            "original_token_count": len(tokens),
            "compressed_token_count": len(pruned_tokens),
            "compression_ratio": round(len(pruned_tokens) / max(1, len(tokens)), 2),
            "burn_rate": round(burn_rate, 3),
            "remaining_budget": round(updated_budget, 3),
            "compressed_prompt": compressed_prompt,
            "output_text": f"[MGE Processed]: Responding to filtered context ({len(pruned_tokens)} tokens)."
        }

# --- Test Execution ---
proxy = MetabolicGatingProxy()
sample_prompt = (
    "warning status sensor alert temperature nominal voltage drop detected "
    "background noise noise noise redundant reading noise system ok telemetry 1002"
)

print("=== TEST 1: HIGH BUDGET (Passive Window Mode) ===")
res_high = proxy.process_request(sample_prompt, energy_budget=90.0, task_context="alert voltage")
print(res_high)

print("\n=== TEST 2: LOW BUDGET (Metabolic Firewall / Survival Mode) ===")
res_low = proxy.process_request(sample_prompt, energy_budget=15.0, task_context="alert voltage")
print(res_low)
