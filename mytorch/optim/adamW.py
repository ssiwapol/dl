import numpy as np

class AdamW():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay=weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # g_W = layer.dLdW + self.weight_decay * layer.W
            # g_b = layer.dLdb + self.weight_decay * layer.b

            # Calculate updates for weight
            self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1-self.beta1) * layer.dLdW
            self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1-self.beta2) * (layer.dLdW ** 2)

            # Calculate updates for bias
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1-self.beta1) * layer.dLdb
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1-self.beta2) * (layer.dLdb ** 2)

            # Perform weight and bias updates with weight decay
            m_W = self.m_W[layer_id] / (1 - (self.beta1 ** self.t))
            v_W = self.v_W[layer_id] / (1 - (self.beta2 ** self.t))
            layer.W = layer.W - self.lr * self.weight_decay * layer.W
            layer.W = layer.W - (self.lr * m_W / np.sqrt(v_W + self.eps))
            m_b = self.m_b[layer_id] / (1 - (self.beta1 ** self.t))
            v_b = self.v_b[layer_id] / (1 - (self.beta2 ** self.t))
            layer.b = layer.b - self.lr * self.weight_decay * layer.b
            layer.b = layer.b - (self.lr * m_b / np.sqrt(v_b + self.eps))
