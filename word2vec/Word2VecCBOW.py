import random
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from .Word2VecAbstract import Word2VecAbstractModel, Word2VecAbstract


class Word2VecCBOWModel(Word2VecAbstractModel):
    """Model for Word2Vec CBOW model."""
    
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int = 300
    ) -> None:
        """
        Initialize the Word2Vec CBOW model.
        args:
            vocab_size: The size of the vocabulary.
            embedding_dim: The dimension of the embedding vectors.
        returns:
            None
        """
        
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

    
    def forward(
        self,
        center_words,
        context_words,
        negative_samples
    ) -> torch.Tensor:
        """
        Forward pass for the Word2Vec CBOW model.
        args:
            center_words: The center words.
            context_words: The context words.
            negative_samples: The negative samples.
        returns:
            The loss.
        """
        
        center_tensor = torch.tensor(center_words, dtype=torch.long)

        context_tensor = [torch.tensor(ctx, dtype=torch.long) for ctx in context_words]
        padded_context_tensor = pad_sequence(context_tensor, batch_first=True, padding_value=0)

        negatives_tensor = torch.tensor(negative_samples, dtype=torch.long)

        v_center = self.output_embeddings(center_tensor)

        v_contexts = self.input_embeddings(padded_context_tensor)
        mask = (padded_context_tensor != 0).unsqueeze(-1) 
        v_contexts = v_contexts * mask.float()
        v_contexts_mean = v_contexts.sum(1) / mask.sum(1).float()

        v_negatives = self.output_embeddings(negatives_tensor)

        loss = Word2VecCBOWModel.calculate_loss(v_contexts_mean, v_center, v_negatives)

        return loss

    
class Word2VecCBOW(Word2VecAbstract):
    """Word2Vec CBOW model."""
    
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

        super().__init__(
            window_radius=window_radius,
            embedding_dim=embedding_dim,
            min_count=min_count
        )


    def set_context_groups_and_model(
        self
    ) -> None:
        """
        Create contextual pairs from data.
        args:
        returns:
            None
        """

        self.model = Word2VecCBOWModel(len(self.word_to_index), self.embedding_dim)
        self.context_groups = []

        for sentence in self.data_tok:
            for i, central_word in enumerate(sentence):
                indicies = range(
                    max(0, i - self.window_radius),
                    min(i + self.window_radius, len(sentence))
                )
                current_group = [
                    self.word_to_index.get(
                        central_word, 
                        self.word_to_index["UNK"]
                    )
                ]
                for context_word_index in indicies:
                    if sentence[context_word_index] == central_word:
                        continue
                    current_group.append(
                        self.word_to_index.get(
                            sentence[context_word_index], 
                            self.word_to_index["UNK"]
                        )
                    )
                
                self.context_groups.append(current_group)

    
    def create_batch(
        self,
        batch_size: int = 32,
        negative_number: int = 10
    ) -> tuple:
        """
        Create a batch of data.
        args:
            batch_size: The size of the batch.
            negative_number: The number of negative samples.
        returns:
            A tuple of center words, context words, and negative samples.
        """
        
        center_batch = []
        context_batch = []
        negative_batch = []

        while len(center_batch) < batch_size:
            center_and_context = random.choice(self.context_groups)

            center_batch.append(center_and_context[0])
            context_batch.append(center_and_context[1:])

            mask = np.ones(self.vocab_size, dtype=bool)
            mask[center_and_context[0]] = False

            neg_candidates = np.arange(self.vocab_size)[mask]
            neg_probs_masked = self.neg_probs_numpy_array[mask]
            neg_probs_masked /= neg_probs_masked.sum()

            negatives = np.random.choice(
                neg_candidates,
                size=negative_number,
                replace=True,
                p=neg_probs_masked
            )

            negative_batch.append(list(negatives))

        return center_batch, context_batch, negative_batch


    def train_model(
        self,
        steps: int = 5,
        batch_size: int = 32,
        negative_number: int = 10
    ) -> None:
        """
        Train the Word2Vec CBOW model.
        args:
            steps: The number of steps to train the model.
            batch_size: The size of the batch.
            negative_number: The number of negative samples.
        returns:
            None
        """
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

        for step in tqdm(range(steps)):
            center_batch, context_batch, negative_batch = self.create_batch(
                batch_size=batch_size,
                negative_number=negative_number
            )

            loss = self.model(center_batch, context_batch, negative_batch)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()

            if step % 1000 == 0:
                print(
                    f"Step {step}, Loss: {loss}, learning rate: {lr_scheduler._last_lr}"
                )
