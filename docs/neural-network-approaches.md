# Neural-Network Approaches

Neural networks learn representations through layers of weighted transformations. They are especially useful for complex nonlinear patterns, spatial data, sequential data, language, and generative modelling.

![Neural network family](images/neural-network-family.svg)

## Architectures covered

| Architecture | Main use | Key idea |
|---|---|---|
| Feedforward neural network | General nonlinear modelling | One-directional mapping from input to output |
| RNN | Sequential data | Uses loops to retain short-term temporal information |
| LSTM | Long sequences | Uses gates to control memory and reduce vanishing-gradient issues |
| CNN | Images and spatial data | Uses convolution, local connections, and weight sharing |
| GAN | Generative modelling | Generator and discriminator trained competitively |
| BERT-style model | Language representation | Bidirectional context representation |
| XLNet-style model | Language representation | Permutation-based autoregressive context modelling |

## BERT-style vs XLNet-style overview

![BERT vs XLNet](images/bert-vs-xlnet.svg)

## Related code

```bash
PYTHONPATH=src python examples/neural_network_demo.py
```

Main module:

```text
src/ai_computational_approaches/neural_networks.py
```
