# History

### Framework Improvements
February 10th, 2025 | By Thomas Breimer | PR approved by Thomas Breimer 
- Can save simulations as videos
- Can decide to run in headless, video, window modes from terminal
- Can run individuals
- Can save individuals to videos
- Added some plots

Feb 11th | By Abhay Kashyap
- Added script to transform CMA-ES output to SNN weights and biases
- Script reshapes the flat array into an array of shape (NUM_SNN * PARAMS_PER_SNN) 
and then returns a list of dictionaries explicitly separating weights and biases of each SNN

### Introducing snn to evogym pipeline
Thurs. Feb 13th | Luodi Wang | SNN output to evogym through .json file
- Saving the output into a json file from a mock (random) input
- Deleted unused files: node.py, layers.py, net.py

### Command Line Args Improvements
February 13th, 2025 | By James Gaskell & Thomas Breimer
- Added proper named command line arguments for all scripts
- Improved implementation with argparse


### John's SNN Code
February 13th, 2025 | By Atharv Tekurkar
- Removed unecessay elements
- Fixed the functions that are giving errors
- Added unit tests for the functions

### Refactoring and Fixes
Feburary 15, 2025 | By Luodi Wang (partnered with Miguel Garduno on pylint)
- `get-output-state()` method in the snn class from a py file
- Within the SNN folder increased pylint score from 5.54/10 to 10/10 by 80.5%!
