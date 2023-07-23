# Adversarial Discriminative Domain Adaptation (ADDA) with a Single Encoder in PyTorch

This repository contains a PyTorch implementation of the Adversarial Discriminative Domain Adaptation ([ADDA](https://arxiv.org/abs/1702.05464)) model, with a modification: it uses a single encoder for both the source and target domains. The model is demonstrated using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset as the source domain and the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset as the target domain, leveraging the ResNet-18 architecture.

## Model Overview
The ADDA model aims to minimize the domain discrepancy between source and target feature representations while maintaining the features' discriminative power for classification. This is achieved by incorporating three types of losses:

1. **Source Classification Loss**: A cross-entropy loss that evaluates the performance of the source encoder and classifier in predicting the source data labels.

2. **Target Encoder Loss**: A domain-adversarial loss that gauges the ability of the target encoder to deceive the discriminator into classifying the target features as if they are from the source domain.

3. **Discriminator Loss**: Another domain-adversarial loss that assesses the proficiency of the discriminator in differentiating between source and target features.

The overall objective of ADDA is to minimize the source classification and target encoder losses while maximizing the discriminator loss.

## Advantages of Using a Single Encoder
My implementation deviates from the original [ADDA](https://arxiv.org/abs/1702.05464) paper by utilizing a single encoder for both domains, yielding several benefits:

- **Computational Efficiency**: Training one encoder instead of two reduces the model's computational demands, resulting in faster training times and lower memory consumption.

- **Shared Feature Learning**: By using a unified encoder, the model is encouraged to learn a common feature representation that is both discriminative for classification and resistant to domain shift. This promotes better generalization to the target domain.

- **Simplicity**: A single-encoder implementation simplifies the model, making it more understandable and easier to maintain.

## Running the Code

You can run the model training script from the command line and specify parameters like the device and number of epochs. Here's how:

1. Open a terminal window.
2. Navigate to the directory containing the script `run_adda.py`.
3. Run the script with your desired parameters. 

For example, to train the model for 100 epochs on the `cuda:0` device, you would run:

```python
python run_adda.py --device cuda:0 --epochs 100
```

In this command:

- `--device` specifies the device to be used for computations. If CUDA is available, you can specify a CUDA device like `cuda:0` or `cuda:1`. If CUDA is not available or you want to use CPU, specify `cpu`.
- `--epochs` specifies the number of epochs to train the model.

By default, if you run the script without any arguments, it will use `cuda:0` as the device and 50 as the number of epochs:

```python
python run_adda.py
```

Please ensure that your Python environment has all the necessary libraries installed, as listed in the `requirements.txt` file.
