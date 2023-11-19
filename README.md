# Experimental Large Language Models Codespace

## Current Experiment
The given Architecture is for a Large Language Model with 64 layers and only about 550 M parameters that allow better loss reduction than current Foundational LLMs. The 64 layers are split in 2 stacks.

* First stack has 32 layers that parallely create about 16 feature dimensions for each token input. 
* Second stack also has 32 layers, parallely creating 64 feature dimensions for each token input.

This leads to a lower memory overhead for Decoder layers, and multiple decoder layers can be parallely stacked together for better outputs. Due to the unavailabiltiy of better hardware cannot test and optimize for 4k, 8K or larger context windows and higher sample size.

Please feel free to take the architecture, train your own LLM model, and share the results with me. Your support and contributions are highly appreciated and will make it better for the commnuity as well.

Note: **This still takes into account all the positional embeddings and attention scores for each token as a normal decoder based model would do**

## Model Architecture

### Overall Model
![Architecture](/img/Architecture.png)

### Decoder Layers
![Decoder](/img/Decoder.png)

## Performance

* **Size** is nearly 1/8th the size of a 32 - 36 Layer 2B normal Transformer Decoder based model at 32 bit precision
* **Cross Entropy Loss** After about 100 steps the current loss stands at 0.37 and reducing. This can be made much better with a larger sample set in range of 16-32 items per batch.
* **Drawbacks** A large Linear layer at the end that increases the memory dependence on RAM due to the increase in number of parameters. 
It is still not completely a retail GPU compatible LLM. The 64 Decoder layers all sit on the GPU leading to a VRAM shortage when training parameters and other overheads are introduced during the training process. Due to this drawback I am only able to train about 2 items per batch only (This could cause issues with Generalization and Loss reduction over varied sample inputs, but without better GPU availability this is the best that I can do on retail as of now.)

## Further study
The model can be improved for 4K and 8K context lengths. For every 2K increase in context will suggest adding another decoder stack in the middle of the Model though this can lead to higher memory requirements

## Current Config
Currently, I have the model configured at
* **Embeddings Dimension**: 512
* **Decoder Layers per stack**: 32
* **Decoder Stacks**: 2
* **Param Count**: 550M
* **Precision Mode**: Single Precision 32 bit
* **Model File Size**: 2.08 GB
* **Tokenizer**: BPE Tokenizer with 110000 vocab size

## Current Training infrastructure
The model is being trained on simple retail specs
* **CPU** - Ryzen 5 5600X
* **GPU** - RTX 3070 8GB
* **RAM** - 32 Gigabyte DDR4
