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

Due Tuesday 02/18
- [ ] build working snn - atharv, jonathan, miguel
    - [x] finish cleaning and implementing basic snn code - node, layer, network
    - figure out how to connect this to the pipeline to get morphology 
    information to get input size, output size, any other info needed
    - [x] figure out how to return a float value for “voxel target length” 
    instead of the 0/1 spike - `if out=0 then target length=0.6; if out=1 then target length=1.6`
    - [ ] visualization of outputs and test results
- [ ] finish unpacking script - abhay
    - [x] transform flat output array into multi-dimensional arrays
    - [x] split into weights & biases for each node for each voxel/snn
    - [ ] test by setting it to dummy SNN and compute
- [ ] pipeline to send and receive data to/from evogym/cmaes scripts - luodi, abhay
    - [ ] extract morphology information from json file relevant to building and initalizing the snn
    - [ ] communicate voxel distance values from evogym and cmaes output from cmaes script to snn, and snn output to evogym