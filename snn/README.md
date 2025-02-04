# Spiking Neural Network

- Neural network that mimics neuron activation and information propagation.
- "Neuron" takes in some input, accumulates over a period of time, activates if current information level is over a threshold (activation = spike value of 1.0). If level is under threshold, neuron is not activated and spike value is 0.0.
- Network is useful when time period consideration is necessary.

## Usage/Testing

`python john_code_snn.py`
- Command runs John's SNN code file, which has a small test written that tests one spiky node and the net as a whole when this file is run.

### Custom Testing
Add custom tests to the block of code at the end of the file, or create a separate file importing `john_code_snn.py` and calling the required classes.

## Designing Custom SNN

- Files `node.py`, `layers.py`, and `net.py` are created for designing our own SNN different from John's. 
- Should be made from scratch using numpy only for math, no use of pytorch/tensorflow.
- Need to figure out the activation function for the SNN that is the most efficient/best for robot actuation.

## Objectives

Due Tuesday 02/11
- [ ] Figure out how to unpack CMA-ES output into weights and biases of individual SNNs - Abhay
- [ ] Build SNN - Atharv, Jonathan, Miguel
- [ ] Build pipeline to get input values from and send output values to the evogym codebase - Luodi
- [ ] Experiment with and test the SNN itself without evogym, then try checking if it works with the pipeline through manually sending custom values - Jonathan, Miguel
- [ ] Compare results using input variations - decimal (actual values) vs binary vs spike-encoded vs normalized - TBD after SNN is built to compare actual results
