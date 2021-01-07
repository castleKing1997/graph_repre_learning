from __future__ import division
from __future__ import absolute_import
import numpy as np
from abc import ABCMeta, abstractmethod
from itertools import combinations_with_replacement
from gensim.models import KeyedVectors
from tqdm import tqdm


class EdgeEmbedder(ABCMeta):

    def __init__(self, keyed_vectors, quiet = False):
        u"""
        :param keyed_vectors: KeyedVectors containing nodes and embeddings to calculate edges for
        """

        self.kv = keyed_vectors
        self.quiet = quiet

    @abstractmethod
    def _embed(self, edge):
        u"""
        Abstract method for implementing the embedding method
        :param edge: tuple of two nodes
        :return: Edge embedding
        """
        pass

    def __getitem__(self, edge):
        if not isinstance(edge, tuple) or not len(edge) == 2:
            raise ValueError(u'edge must be a tuple of two nodes')

        if edge[0] not in self.kv.index2word:
            raise KeyError(u'node {} does not exist in given KeyedVectors'.format(edge[0]))

        if edge[1] not in self.kv.index2word:
            raise KeyError(u'node {} does not exist in given KeyedVectors'.format(edge[1]))

        return self._embed(edge)

    def as_keyed_vectors(self):
        u"""
        Generated a KeyedVectors instance with all the possible edge embeddings
        :return: Edge embeddings
        """

        edge_generator = combinations_with_replacement(self.kv.index2word, r=2)

        if not self.quiet:
            vocab_size = len(self.kv.vocab)
            total_size = reduce(lambda x, y: x * y, xrange(1, vocab_size + 2)) / \
                         (2 * reduce(lambda x, y: x * y, xrange(1, vocab_size)))

            edge_generator = tqdm(edge_generator, desc=u'Generating edge features', total=total_size)

        # Generate features
        tokens = []
        features = []
        for edge in edge_generator:
            token = unicode(tuple(sorted(edge)))
            embedding = self._embed(edge)

            tokens.append(token)
            features.append(embedding)

        # Build KV instance
        edge_kv = KeyedVectors(vector_size=self.kv.vector_size)
        edge_kv.add(
            entities=tokens,
            weights=features)

        return edge_kv


class AverageEmbedder(EdgeEmbedder):
    u"""
    Average node features
    """

    def _embed(self, edge):
        return (self.kv[edge[0]] + self.kv[edge[1]]) / 2


class HadamardEmbedder(EdgeEmbedder):
    u"""
    Hadamard product node features
    """

    def _embed(self, edge):
        return self.kv[edge[0]] * self.kv[edge[1]]


class WeightedL1Embedder(EdgeEmbedder):
    u"""
    Weighted L1 node features
    """

    def _embed(self, edge):
        return np.abs(self.kv[edge[0]] - self.kv[edge[1]])


class WeightedL2Embedder(EdgeEmbedder):
    u"""
    Weighted L2 node features
    """

    def _embed(self, edge):
        return (self.kv[edge[0]] - self.kv[edge[1]]) ** 2
