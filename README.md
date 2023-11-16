# Experimental Models Codespace

## Current Experiment
The given Architecture is for a Large Language Model with 32 layers and only about 780M parameters that allow better loss reduction than current Foundational LLMs. The 128 layers are split in 2 stacks. First stack has 64 layers parallely create about 8 feature dimensions each for the tokens. Last stack has 64 layers as well parallely creating 16 feature dimensions each for the tokens. This leads to a lower memory overhead for Decoder layers, and multiple decoder layers can be parallely stacked together for better outputs. Due to the unavailabiltiy of better hardware cannot test and optimize for 4k, 8K or larger context windows and higher sample size. Please feel free to take the architecture and train your own LLM model. There are tons of pretraining datasets available on the internet, use any and check the performance for yourself.

Note: **This still takes into account all the positional embeddings and attention scores for each token as a normal decoder based model would do**

## Model Architecture

### Overall Model
![Architecture](/img/Architecture.png)

### Decoder Layers
![Decoder](/img/Decoder.png)

## Performance

* **Size** is nearly 1/8th the size of a 32 - 36 Layer 2B normal Transformer Decoder based model at 32 bit precision
* **Cross Entropy Loss** After 2700+ steps the current loss stands at 0.07 and reducing
* **Drawbacks** A large Linear layer at the end that increases the memory dependence due to the increase in number of parameters. This can be bad for the GPU

## Further study
The model can be improved for 4K and 8K context lengths. For every 2K increase in context will suggest adding another decoder stack in the middle of the Model though this can lead to higher memory requirements

## Current Config
Currently, I have the model configured at
* **Embeddings Dimension**: 512
* **Decoder Layers per stack**: 64
* **Decoder Stacks**: 2
* **Param Count**: 450M
* **Precision Mode**: Mixed Precision (32 bit)
* **Model File Size**: 1.70 GB
* **Tokenizer**: BPE Tokenizer with 110000 vocab size

## Current Training infrastructure
The model is being trained on simple retail specs
* **CPU** - Ryzen 5 5600X
* **GPU** - RTX 3070 8GB
* **RAM** - 32 Gigabyte DDR4
