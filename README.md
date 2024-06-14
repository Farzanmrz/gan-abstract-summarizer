# Project Overview

This project implements a text generation and classification pipeline using the BART model for conditional text generation and a custom CNN-based Discriminator for text classification. The objective is to generate summaries for articles and classify them as either machine-generated or human-generated. This README document provides detailed instructions on how to set up, run, and understand the project, ensuring it can be taken up for continued development or analysis.

## Project Structure

- `discriminator.py`: Defines the `Discriminator` class, a CNN-based neural network for classifying summaries.
- `main.py`: Contains the main pipeline for loading data, training the generator and discriminator, and evaluating the performance.
- `generator_checkpoint.pth`: Stores the state of the generator model.
- `discriminator_checkpoint.pth`: Stores the state of the discriminator model.
- `first_article_summaries.txt`: Holds the original and generated summaries of the first article for each epoch.

## Requirements

To run this project, the following packages are required:
- `torch`
- `transformers`
- `datasets`
- `evaluate`

You can install these packages using pip:

```bash
pip install torch transformers datasets evaluate
```

## File Descriptions

### discriminator.py

Defines the `Discriminator` class, which is a convolutional neural network (CNN) for classifying summaries.

- **Class: Discriminator**
  - `__init__(self, vocab_size, embed_size, num_classes=2)`: Initializes the layers of the discriminator.
  - `conv_and_pool(self, x, conv)`: Applies convolution and max pooling operations.
  - `forward(self, x)`: Defines the forward pass through the network.

### main.py

Contains the main script to load the dataset, initialize models, train, and evaluate the generator and discriminator.

- **Load ROUGE Metric**
  - `rouge = evaluate.load('rouge')`: Loads the ROUGE metric for evaluation.

- **Load Dataset**
  - `dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:20]")`: Loads a subset of the CNN/Daily Mail dataset.

- **Split Dataset**
  - Splits the dataset into training, validation, and test sets.

- **Initialize BART Model and Tokenizer**
  - `tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')`
  - `generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')`

- **Initialize Discriminator**
  - `discriminator = Discriminator(vocab_size, embed_size)`

- **Training Parameters**
  - Defines loss functions, optimizers, and training parameters.

- **Load Model Checkpoints**
  - Loads saved model states if they exist:
    - `generator_checkpoint.pth`: Stores the state of the generator model.
    - `discriminator_checkpoint.pth`: Stores the state of the discriminator model.

- **Training Loop**
  - **Train Discriminator**: 
    - `train_discriminator()`: Trains the discriminator on human and machine-generated summaries.
  - **Train Generator**: 
    - `train_generator()`: Trains the generator using policy gradient and maximum likelihood losses.
  - **Evaluate Generator**: 
    - `evaluate_generator(epoch)`: Evaluates the generator using ROUGE scores and saves summaries.

- **Save and Load Functions**
  - Functions to save the original summary and model checkpoints.
  - `first_article_summaries.txt`: Holds the original and generated summaries of the first article for each epoch.

## Running the Project

1. **Load Dataset**: The dataset is loaded and split into training, validation, and test sets.
2. **Initialize Models**: The BART model and tokenizer are initialized, followed by the discriminator.
3. **Train Models**: The training loop iterates over epochs, training the generator and discriminator, and evaluating the performance.
4. **Evaluate Models**: The generator is evaluated using the ROUGE metric, and results are printed and saved.

```bash
python main.py
```
