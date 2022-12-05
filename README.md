# RNN from scratch

An implementation of the Recurrent Neural Network architecture from the ground up using [NumPy](https://github.com/numpy/numpy).

The model is trained in a simple alphabet sequence, for demonstration purposes. The back propagation through time algorithm was implemented with a gradient descent optimizer and negative log likelihood loss function hardcoded, as this project serves educational purposes. By no means should it be used in production.

During the training phase, the model state with the least loss and highest accuracy is saved, which is then used for inference. A pretrained model is contained in the repository, which can be used similarly to the inference part of the [train.py](./train.py) file.
