# Word2Vec Implementation

A PyTorch implementation of Word2Vec algorithms including both Continuous Bag of Words (CBOW) and Skip-Gram models with negative sampling and subsampling techniques.

## Overview

This project provides a complete implementation of the Word2Vec algorithm as described in the original paper "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. The implementation includes:

- **CBOW (Continuous Bag of Words)**: Predicts the center word from context words
- **Skip-Gram**: Predicts context words from the center word
- **Negative Sampling**: Efficient training technique to avoid expensive softmax computation
- **Subsampling**: Reduces the influence of frequent words during training

## Project Structure

```
Word2Vec/
├── word2vec/
│   ├── Word2VecAbstract.py      # Abstract base classes and common functionality
│   ├── Word2VecCBOW.py          # CBOW model implementation
│   └── Word2VecSkipGram.py      # Skip-Gram model implementation
├── data/
│   └── quora.txt                # Sample dataset (Quora questions)
├── examples.ipynb               # Jupyter notebook with usage examples
└── README.md                    # This file
```

## Usage

### Basic Usage

```python
from word2vec.Word2VecCBOW import Word2VecCBOW
from word2vec.Word2VecSkipGram import Word2VecSkipGram

# Load your text data
data = ["Your text data here", "More text data", ...]

# CBOW Model
cbow = Word2VecCBOW(window_radius=5, embedding_dim=300, min_count=5)
cbow.set_text_before_context_pairs(data)
cbow.set_context_groups_and_model()
cbow.subsampling_probabilities()
cbow.negative_sampling_probabilities()
cbow.train_model(steps=10000, batch_size=128, negative_number=15)

# Skip-Gram Model
skipgram = Word2VecSkipGram(window_radius=5, embedding_dim=300, min_count=5)
skipgram.set_text_before_context_pairs(data)
skipgram.set_context_groups_and_model()
skipgram.subsampling_probabilities()
skipgram.negative_sampling_probabilities()
skipgram.train_model(steps=10000, batch_size=128, negative_number=15)
```

### Finding Similar Words

```python
# Get word embeddings
word_embeddings = cbow.get_center_embeddings_by_words(["python"])

# Find nearest neighbors (example function from notebook)
def find_nearest(model, words):
    word_vector = model.get_center_embeddings_by_words(words)
    # Implementation details in examples.ipynb
    return nearest_words
```

## Key Classes

### Word2VecAbstract
Base class containing common functionality:
- Text preprocessing and tokenization
- Vocabulary building with minimum count filtering
- Subsampling probability calculation
- Negative sampling probability calculation

### Word2VecCBOW
CBOW model implementation:
- Predicts center word from context words
- Handles variable-length context windows
- Uses padding for batch processing

### Word2VecSkipGram
Skip-Gram model implementation:
- Predicts context words from center word
- Implements subsampling during training
- More efficient for large vocabularies

## Hyperparameters

- `window_radius`: Size of the context window (default: 5)
- `embedding_dim`: Dimension of word embeddings (default: 300)
- `min_count`: Minimum word frequency to include in vocabulary (default: 5)
- `subsampling_threshold`: Threshold for subsampling (default: 1e-3)
- `negative_number`: Number of negative samples per positive sample (default: 10)
- `batch_size`: Training batch size (default: 32)

## Training Details

The implementation uses:
- **Adam optimizer** with learning rate 1e-3
- **Learning rate scheduling** (reduced by 0.5 every 5000 steps)
- **Gradient clipping** (max norm: 1.0)
- **Negative sampling** for efficient training
- **Subsampling** to reduce frequent word influence

## Example Results

The included Jupyter notebook (`examples.ipynb`) demonstrates training on Quora questions data and shows that the model successfully learns semantic relationships between words (e.g., "python" is associated with other programming languages).

## Mathematical Background

### Loss Function
The implementation uses the standard Word2Vec loss with negative sampling:


$$Loss = -\log{\sigma(u_i · v_j)} - \sum^{k}_{r=1} \log{\sigma(-u_i · v_r)}$$


Where:
- $u_i$ is the center word embedding
- $v_j$ is the positive context word embedding  
- $v_r$ are negative sample embeddings
- $\sigma$ is the sigmoid function
- $k$ is number of negative words

### Subsampling
Words are randomly dropped during training with probability:

$$P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$


Where `t` is the subsampling threshold and $f(w_i)$ is the word frequency.

### Negative Sampling
Negative samples are drawn with probability:

$$P(w_i) = \frac{f(w_i)^{(3/4)}}{Z}$$


Where `Z` is a normalization constant.
