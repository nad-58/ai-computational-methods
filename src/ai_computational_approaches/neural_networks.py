"""Neural-network examples and architecture summaries."""
import numpy as np


class TinyFeedForwardNetwork:
    """A tiny NumPy feedforward neural network for XOR-style learning."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.5, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, 0.5, size=(hidden_dim, 1))
        self.b2 = np.zeros(1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        return self.sigmoid(h @ self.w2 + self.b2)

    def fit(self, x, y, epochs: int = 300, lr: float = 0.1) -> None:
        y = y.reshape(-1, 1)
        n = len(x)
        for _ in range(epochs):
            h = np.tanh(x @ self.w1 + self.b1)
            y_hat = self.sigmoid(h @ self.w2 + self.b2)
            dy = (y_hat - y) / n
            self.w2 -= lr * (h.T @ dy)
            self.b2 -= lr * dy.sum(axis=0)
            dh = dy @ self.w2.T * (1 - h**2)
            self.w1 -= lr * (x.T @ dh)
            self.b1 -= lr * dh.sum(axis=0)

    def predict(self, x):
        return (self.forward(x) >= 0.5).astype(int).ravel()


def feedforward_demo() -> dict:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = TinyFeedForwardNetwork()
    model.fit(x, y)
    pred = model.predict(x)
    return {"method": "feedforward_neural_network", "xor_predictions": pred.tolist(), "accuracy": float((pred == y).mean())}


def architecture_summary() -> dict:
    return {
        "ffnn": "one-directional feature transformation and prediction",
        "rnn": "sequence modelling with loop-based short-term memory",
        "lstm": "gated recurrent model for long-term dependencies",
        "cnn": "spatial feature extraction with convolution and weight sharing",
        "gan": "generator and discriminator trained competitively",
        "bert_style": "bidirectional contextual representation using masked-token style pretraining",
        "xlnet_style": "permutation-based autoregressive contextual representation",
    }


if __name__ == "__main__":
    print(feedforward_demo())
    print(architecture_summary())
