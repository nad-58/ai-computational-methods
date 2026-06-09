"""Small transparent numerical examples for neural-network families."""
from __future__ import annotations

import math

import numpy as np


def _round(values, digits: int = 4):
    return np.asarray(values).round(digits).tolist()


def feedforward_example() -> dict:
    x = np.array([0.6, -0.2])
    w1 = np.array([[0.5, -0.3], [0.8, 0.2]])
    b1 = np.array([0.1, -0.1])
    w2 = np.array([[0.7], [-0.4]])
    b2 = np.array([0.05])
    hidden_linear = x @ w1 + b1
    hidden = np.tanh(hidden_linear)
    output_linear = hidden @ w2 + b2
    probability = 1 / (1 + np.exp(-output_linear))
    return {
        "input": x.tolist(),
        "hidden_linear": _round(hidden_linear),
        "hidden_activation": _round(hidden),
        "output_linear": round(float(output_linear[0]), 4),
        "probability": round(float(probability[0]), 4),
        "prediction": int(probability[0] >= 0.5),
        "flow": "2 inputs -> 2 hidden neurons -> sigmoid output",
    }


def rnn_example() -> dict:
    sequence = [0.2, 0.5, -0.1, 0.7]
    wx, wh, bias = 0.8, 0.4, 0.1
    hidden = 0.0
    states = []
    for value in sequence:
        hidden = math.tanh(wx * value + wh * hidden + bias)
        states.append(round(hidden, 4))
    return {
        "sequence": sequence,
        "weights": {"input": wx, "recurrent": wh, "bias": bias},
        "hidden_states": states,
        "final_state": states[-1],
        "flow": "x_t + previous hidden state -> tanh -> new hidden state",
    }


def lstm_example() -> dict:
    sequence = [0.4, -0.2, 0.6]
    hidden, cell = 0.0, 0.0
    trace = []

    def sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))

    for value in sequence:
        forget = sigmoid(0.7 * value + 0.2 * hidden + 0.1)
        input_gate = sigmoid(0.6 * value - 0.1 * hidden)
        output_gate = sigmoid(0.5 * value + 0.1 * hidden)
        candidate = math.tanh(0.9 * value + 0.3 * hidden)
        cell = forget * cell + input_gate * candidate
        hidden = output_gate * math.tanh(cell)
        trace.append(
            {
                "input": value,
                "forget_gate": round(forget, 4),
                "input_gate": round(input_gate, 4),
                "output_gate": round(output_gate, 4),
                "candidate": round(candidate, 4),
                "cell_state": round(cell, 4),
                "hidden_state": round(hidden, 4),
            }
        )
    return {
        "steps": trace,
        "flow": "input -> forget/input/output gates -> cell memory -> hidden output",
    }


def cnn_example() -> dict:
    image = np.array(
        [[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
        dtype=float,
    )
    kernel = np.array([[1, 0], [0, -1]], dtype=float)
    feature_map = np.zeros((3, 3))
    for row in range(3):
        for column in range(3):
            patch = image[row : row + 2, column : column + 2]
            feature_map[row, column] = float(np.sum(patch * kernel))
    return {
        "image": image.astype(int).tolist(),
        "kernel": kernel.astype(int).tolist(),
        "feature_map": feature_map.tolist(),
        "max_activation": float(feature_map.max()),
        "flow": "4x4 image -> sliding 2x2 kernel -> 3x3 feature map",
    }


def gan_example() -> dict:
    noise = np.array([0.3, -0.5])
    generator_weights = np.array([[0.8, -0.2], [0.4, 0.6]])
    generated_sample = np.tanh(noise @ generator_weights)
    discriminator_weights = np.array([1.2, -0.7])
    logit = float(generated_sample @ discriminator_weights)
    probability_real = 1 / (1 + math.exp(-logit))
    generator_loss = -math.log(max(probability_real, 1e-12))
    return {
        "noise": noise.tolist(),
        "generator_weights": generator_weights.tolist(),
        "generated_sample": _round(generated_sample),
        "discriminator_logit": round(logit, 4),
        "probability_real": round(probability_real, 4),
        "generator_loss": round(generator_loss, 4),
        "flow": "noise -> generator -> synthetic sample -> discriminator -> adversarial loss",
    }


def bert_style_example() -> dict:
    tokens = ["AI", "uses", "data"]
    embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    scores = embeddings @ embeddings.T / math.sqrt(2)
    shifted = scores - scores.max(axis=1, keepdims=True)
    attention = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    contextual = attention @ embeddings
    return {
        "tokens": tokens,
        "embeddings": embeddings.tolist(),
        "attention_scores": _round(scores),
        "attention_weights": _round(attention),
        "contextual_embeddings": _round(contextual),
        "flow": "all tokens -> bidirectional self-attention -> contextual representations",
    }


def xlnet_style_example() -> dict:
    token_values = {"A": 1.0, "B": 2.0, "C": 3.0}
    permutations = [["A", "B", "C"], ["C", "A", "B"]]
    traces = []
    for order in permutations:
        running_context = 0.0
        steps = []
        for token in order:
            prediction_score = 0.6 * running_context + 0.4 * token_values[token]
            steps.append(
                {
                    "token": token,
                    "context_before": round(running_context, 4),
                    "prediction_score": round(prediction_score, 4),
                }
            )
            running_context += token_values[token]
        traces.append({"order": order, "steps": steps})
    return {
        "token_values": token_values,
        "permutations": traces,
        "flow": "permuted token order -> autoregressive context -> prediction score",
    }


def all_neural_examples() -> dict:
    return {
        "feedforward": feedforward_example(),
        "rnn": rnn_example(),
        "lstm": lstm_example(),
        "cnn": cnn_example(),
        "gan": gan_example(),
        "bert_style": bert_style_example(),
        "xlnet_style": xlnet_style_example(),
    }
