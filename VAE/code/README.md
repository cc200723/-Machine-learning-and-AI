# COMP7043 Assignment 3

In Assignment 3, your will implement VAE, and run experiments using the MNIST dataset.

You can download the datasets and extract them to the `data/` directory in the root of this repository). Code for loading and processing this data into minibatches are provided in `mnist.py`.

Scaffolding for your VAE models is provided in `models.py`, and you will implement the loss functions for various types of Gaussian VAE's in `losses.py`.

Framework code for training your models is found in `gaussian_vae.ipynb`. It is recommended to use Google Colab or Kaggle to execute the notebook (with a GPU accelerator attached). For debugging, you may find it helpful to modify the default hyper-parameters to build smaller models that are faster to train.

# File that need to submit
- Implemented `losses.py` file.
- Implemented `models.py` file.
- A short report explaining your model and loss implementation and discuss the generation result.