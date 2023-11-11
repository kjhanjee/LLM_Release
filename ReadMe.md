# Experimental Models Codespace

## Current Experiment
Currently Architecture is for a Large Language Model with 128 layers and only about 400M parameters that allow better loss reduction that current Foundational LLMs. The 128 layers are split in 2 stacks. First stack has 64 layers parallely create about 8 features for the tokens. Second stack has 64 layers as well parallely creating 32 features for the tokens. This leads to a lower memory overhead for Decoder layers, and multiple decoder layers can be parallely stacked together for better outputs. Due to the unavailabiltiy of better hardware cannot test and optimize for 4k, 8K or larger context windows and higher sample size. Please feel free to take the architecture and train your own LLM model. There are tons of pretraining datasets available on the internet, use any and check the performance for yourself.

Note: **This still takes into account all the positional embeddings and attention scores for each token as a normal decoder based model would do**

## Model Architecture
![Architecture](/img/Architecture.png)

## Performance

### Size
Model is nearly 1/10th the size of a 128 Layer normal Transformer Decoder based model 

### Loss Reduction
Model seems to be getting low losses after just 100 training steps which given the model size and architecture is a great feat (IMO).

### Drawbacks 
Model has a large Linear layer at the end that increases the memory dependence due to the increase in number of parameters. This can be bad for the GPU

## Further study
The model can be improved for 4K and 8K context lengths. For every 2K increase in context will suggest adding another decoder stack in the middle of the Model though this can lead to higher memory requirements

## Current Training infrastructure
Currently the model is being trained on simple retail specs
* CPU - Ryzen 5 5600X
* GPU - RTX 3070 8GB
* RAM - 32 Gigabyte DDR4
