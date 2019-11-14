
# Modified from tutorial of Pytorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch.nn as nn
import torch.nn.modules.transformer as T
from .positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, num_of_vocab, num_of_head, num_of_layer, dim_word, dim_hidden, dropout=0.5):
        '''
        :param
        : num_of_vocab: size of vocabulary list
        : num_of_head:  number of attention head
        : num_of_layer: number of
        : dim_word:
        : dim_hidden:
        '''
        super().__init__()
        self.model_type = 'Transformer'
        self.dim_word = dim_word
        self.embeddings = nn.Embedding(num_of_vocab, dim_word)
        self.positional_encoder = PositionalEncoding(d_model=dim_word)
        self.encoder = T.TransformerEncoder(
            encoder_layer=T.TransformerEncoderLayer(d_model=dim_word,
                                                    nhead=num_of_head,
                                                    dim_feedforward=dim_hidden,
                                                    dropout=dropout),
            num_layers=num_of_layer)
        self.decoder = nn.Linear(dim_word, num_of_vocab)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # FIXME
        src = self.embeddings(src) * math.sqrt(self.dim_word)
        src = self.positional_encoder(src)
        output = self.encoder(src)
        output = self.decoder(output)
        return output

