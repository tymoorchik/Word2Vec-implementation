
from abc import ABC, abstractmethod
import string
from typing import List

from math import sqrt

from torch.nn import init
from torch import nn
from torch.nn import functional as F
import torch

from nltk.tokenize import WordPunctTokenizer
from collections import Counter
import numpy as np


class Word2VecAbstractModel(nn.Module):
    """Abstract class for Word2Vec model. Will be inherited by SkipGram and CBOW models."""

    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int = 300
    ) -> None:
        """
        Initialize the Word2Vec model.
        args:
            vocab_size: The size of the vocabulary.
            embedding_dim: The dimension of the embedding vectors.
        returns:
            None
        """

        super().__init__()

        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        initrange = 1.0 / sqrt(embedding_dim)
        init.uniform_(self.input_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.output_embeddings.weight.data, -initrange, initrange)

    @staticmethod
    def calculate_loss(
        main_embeddings: List[float],
        positive_embddings: List[float],
        negative_samples_embeddings: List[float]
    ) -> None:
        """
        Calculate the loss for the Word2Vec model.
        Classic loss: log-sigmoid(u_i * v_j) + sum(log-sigmoid(-u_i * v_k)) for k in [1, ..., K]
        args:
            main_embeddings: The embeddings of the main words.
            positive_embddings: The embeddings of the positive words.
            negative_samples_embeddings: The embeddings of the negative words.
        returns:
            The loss.
        """

        pos_score = torch.sum(main_embeddings * positive_embddings, dim=1)
        pos_loss = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(negative_samples_embeddings, main_embeddings.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        loss = (pos_loss + neg_loss).mean()

        return loss

    
    @abstractmethod
    def forward(
        self, 
        *args, 
        **kwargs
    ) -> None:
        """
        Forward raise error for the Word2Vec model.
        args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        returns:
            None
        """
        raise NotImplementedError("Forward pass is not implemented for the Word2Vec model.")


class Word2VecAbstract:
    """Abstract class for Word2Vec model. Will be inherited by SkipGram and CBOW models."""
    def __init__(
        self, 
        window_radius: int = 5, 
        embedding_dim: int = 300, 
        min_count: int = 5,
    ) -> None:
        """
        Initialize the Word2Vec model.
        args:
            window_radius: The radius of the window for the context words.
            embedding_dim: The dimension of the embedding vectors.
            min_count: The minimum number of times a word must appear in the data to be included in the vocabulary.
        returns:
            None
        """

        self.model = None
        self.window_radius = window_radius
        self.embedding_dim = embedding_dim
        self.min_count = min_count

    
    def set_text_before_context_pairs(
        self, 
        data: List[str]
    ) -> None:
        """
        Set the data for the model: tokenize and clean it. Set vocabularry and counter
        args:
            data: text for training
        returns:
            None
        """

        tokenizer = WordPunctTokenizer()

        data_tok = [
            tokenizer.tokenize(
                line.translate(str.maketrans("", "", string.punctuation)).lower()
            )
            for line in data
        ]
        data_tok = [x for x in data_tok if len(x) >= 3]

        vocabulary_counter = Counter([word for s in data_tok for word in s])

        word_count_dict = dict()
        for word, counter in vocabulary_counter.items():
            if counter >= self.min_count:
                word_count_dict[word] = counter
            else:
                if 'UNK' in word_count_dict:
                    word_count_dict['UNK'] += counter
                else:
                    word_count_dict['UNK'] = counter

        if not ("UNK" in word_count_dict):
            word_count_dict["UNK"] = 1
            
        vocabulary = set(word_count_dict.keys())

        self.data_tok = data_tok
        self.vocabulary = vocabulary
        self.word_count_dict =  word_count_dict
        self.word_to_index = {word: index for index, word in enumerate(vocabulary)}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.vocab_size = len(self.word_to_index)

    
    def subsampling_probabilities(
        self, 
        subsampling_threshold: float = 1e-3
    ) -> None:
        """
        Probability to keep a word in training: 
        P_keep = max(0, 1 - sqrt(threshold / frequency))
        args:
            subsampling_threshold: The threshold for the subsampling.
        returns:
            None
        """
        self.word_subsampling_probabilities_dict = {}
        self.subsampling_threshold = subsampling_threshold

        norm = sum(self.word_count_dict.values())
        for word, count in self.word_count_dict.items():
            freq = count / norm
            
            del_prob = max(0.0, 1 - sqrt(subsampling_threshold / freq))
            self.word_subsampling_probabilities_dict[word] = del_prob


    def negative_sampling_probabilities(
        self
    ) -> None:
        """
        Calculating dictionary with probability to use word in training as negative sample.
        Formule: p_[neg_use](w) = (frequency(w) / sum( all words count )) ** (3 / 4) / Z
            Z - norming
        args:
        returns:
            None
        """

        self.word_negative_sampling_probabilities_dict = {}

        sum_all = sum(x[1] for x in self.word_count_dict.items())
        sum_unigram = sum((count / sum_all)**(0.75) for _, count in self.word_count_dict.items())

        for word, count in self.word_count_dict.items():
            self.word_negative_sampling_probabilities_dict[word] = (count / sum_all)**(0.75) / sum_unigram

        self.neg_probs_numpy_array = np.array([
            self.word_negative_sampling_probabilities_dict[self.index_to_word[idx]]
            for idx in range(self.vocab_size)
        ])


    def get_center_embeddings_by_words(
        self,
        words
    ) -> None:
        """
        Get the center embeddings for the words.
        args:
            words: The words to get the center embeddings for.
        returns:
            The center embeddings for the words.
        """

        center_words = [self.word_to_index[word] for word in words]
        return self.model.input_embeddings(torch.tensor(center_words, requires_grad=False).long()).detach()

    
    def get_context_embeddings_by_words(
        self,
        words
    ) -> None:
        """
        Get the context embeddings for the words.
        args:
            words: The words to get the context embeddings for.
        returns:
            The context embeddings for the words.
        """
        
        context_words = [self.word_to_index[word] for word in words]
        return self.model.output_embeddings(torch.tensor(context_words, requires_grad=False).long()).detach()
