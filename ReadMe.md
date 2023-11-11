# Experimental Models Codespace

## Current Experiment
Currently experimenting on creating model with 64 parallel layers in two stacks connected serially

## Model Architecture
![Architecture](/img/Architecture.png)

## Performance

### Size
Model is nearly 1/5th the size of a 128 Layer normal Transformer Decoder based model 

### Loss Reduction
Model seems to be getting low losses after just 100 training steps which given the model size and architecture is a great feat

### Drawbacks 
Model has a large Linear layer at the end that increases the memory dependence due to the increase in number of parameters

## Further study
The model can be improved for 4K and 8K context lengths. For every 2K increase in context will suggest adding another decoder stack in the middle of the Model though this can lead to higher memory requirements

## Current Training infrastructure
Currently the model is being trained on simple retail specs
CPU - Ryzen 5 5600X
GPU - RTX 3070 8GB
RAM - 32 Gigabyte DDR4
