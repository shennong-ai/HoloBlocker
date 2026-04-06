# holoblocker_minimal.py
import numpy as np
from sentence_transformers import SentenceTransformer

class MinimalHoloBlocker:
    def __init__(self, tau=0.5, conf_threshold=0.3):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nodes = ["dirty hands", "mouth", "feed", "polite", "normal"]
        self.node_vecs = np.array([self.encoder.encode(n) for n in self.nodes])
        # Predefined Laplacian (for illustration)
        self.L = np.array([[ 1.0, -0.9,  0.0,  0.0,  0.0],
                           [-0.9,  1.0, -0.9,  0.0,  0.0],
                           [ 0.0, -0.9,  1.0,  0.0,  0.0],
                           [ 0.0,  0.0,  0.0,  1.0, -0.5],
                           [ 0.0,  0.0,  0.0, -0.5,  1.0]])
        self.tau = tau
        self.conf_threshold = conf_threshold

    def check(self, text):
        vec = self.encoder.encode(text)
        x = np.array([np.dot(vec, v) / (np.linalg.norm(vec)*np.linalg.norm(v)) for v in self.node_vecs])
        confidence = np.max(x)
        if confidence < self.conf_threshold:
            return False, f"OOD (conf={confidence:.2f})"
        x_norm = x / np.linalg.norm(x)
        energy = x_norm @ self.L @ x_norm
        if energy > self.tau:
            return False, f"Topology energy={energy:.2f}"
        return True, "safe"

if __name__ == "__main__":
    hb = MinimalHoloBlocker()
    tests = ["feed the elderly", "dirty hands feed", "asdfghjkl"]
    for t in tests:
        safe, reason = hb.check(t)
        print(f"{t:30} -> {'✅' if safe else '❌'} {reason}")