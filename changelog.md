# V 0.0.2a

## Changes

### Architectural
* *Decoupled Multi Head Attention from Decoder*: Decoupling the attention head layer from the decoder layers allow additional layers to be added as Attention mechanism might not be necessary for each decoder layer but for the whole decoder stack
* *Added Additional Decoder Stacks*: Adding Two additional Decoder stacks lead to an increase in model parameters which will be better for language comprehension
* *Increased Feed Forward In Decoder Stack to 8 times the embedding size*: Increasing the Feed Forward dimension in the Decoder stack allow additional features to be present during the Generating process and also for final linear layer for token probability prediction
* *Added half precision training for Linar layers*: Introduced Half Precision training for the Linear layers significantly reducing the memory requirements and allowing addition of another layer

### Training
* *Batch Size*: Increased batch size to 3, and 1024 token iteration between each item for better model generalization
* *SGD Optimizer*: Switched Optimizer to SGD instead of ADAM to prevent nan loss and gradients

### Metrics
* *Perplexity*: Added Perplexity calculation during training prints. This will help keep the perplexity monitoring in action during training

## Future Outlook
* Will continue to monitor the model loss reduction and perplexity reduction
* Try to figure out a way to chart out losses and Perplexity in a dynamic interactive charts without increasing memory load
