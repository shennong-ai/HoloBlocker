# 在 HoloBlockerFull 类中添加/修改以下方法

def project_text(self, text: str) -> Tuple[np.ndarray, float]:
    """
    返回 (原始相似度向量, 最大置信度)
    不再做归一化，保留原始相似度用于置信度判断。
    """
    text_vec = self.encoder.encode(text)
    raw_x = np.array([
        cosine_similarity([text_vec], [self.node_vectors[i]])[0][0]
        for i in range(self.n_nodes)
    ])
    # 置信度：最大相似度（也可用均值，推荐最大值）
    confidence = np.max(raw_x)
    return raw_x, confidence

def compute_energy(self, x: np.ndarray) -> float:
    """输入原始相似度向量，内部归一化后计算能量"""
    norm = np.linalg.norm(x)
    if norm < 1e-6:
        return 0.0
    x_norm = x / norm
    return float(x_norm @ self.L @ x_norm)

def check(self, text: str, conf_threshold: float = 0.3) -> Tuple[bool, Dict]:
    raw_x, confidence = self.project_text(text)
    
    # 第一道防线：置信度过低（听不懂）
    if confidence < conf_threshold:
        is_safe = False
        energy = 0.0
        trigger_reason = f"OOD (confidence={confidence:.3f} < {conf_threshold})"
    else:
        # 第二道防线：拓扑禁忌
        energy = self.compute_energy(raw_x)
        is_safe = energy < self.tau
        trigger_reason = f"Topology energy={energy:.3f} >= {self.tau}" if not is_safe else "Safe"
    
    # 找激活最高的节点（用于展示）
    top_indices = np.argsort(raw_x)[-3:][::-1]
    top_nodes = [(self.nodes[i], float(raw_x[i])) for i in top_indices if raw_x[i] > 0.1]
    
    report = {
        "confidence": confidence,
        "conf_threshold": conf_threshold,
        "energy": energy,
        "threshold": self.tau,
        "is_safe": is_safe,
        "top_nodes": top_nodes,
        "trigger_reason": trigger_reason,
        "raw_x": raw_x,
    }
    return is_safe, report