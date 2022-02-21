import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    
    emb_model = np.load(emb_file ,allow_pickle=True).item()
    
    emb_matrix = []
    for word in vocab.word2id.keys():
        if emb_model.get(word) is not None:
            emb_matrix.append(emb_model.get(word))
        else:
            emb_matrix.append(np.random.random(emb_size))
            
    return np.array(emb_matrix)


    # raise NotImplementedError()


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embed = nn.Embedding(len(self.vocab), self.args.emb_size)
        # self.emb_drop = nn.Dropout(self.args.emb_drop)
        # self.emb_bn = nn.BatchNorm1d(self.args.emb_size)

        fully_connected_layers = []
        for i in range(self.args.hid_layer):
            if i == 0:
                fully_connected_layers.append(nn.Linear(self.args.emb_size, self.args.hid_size))
                fully_connected_layers.append(nn.Dropout(self.args.hid_drop))
                # fully_connected_layers.append(nn.BatchNorm1d(self.args.hid_size))
            elif i == self.args.hid_layer - 1:
                fully_connected_layers.append(nn.Linear(self.args.hid_size, self.tag_size))
            else:
                fully_connected_layers.append(nn.Linear(self.args.hid_size, self.args.hid_size))
                fully_connected_layers.append(nn.Dropout(self.args.hid_drop))
                #fully_connected_layers.append(nn.BatchNorm1d(self.args.hid_size))

        self.fc = nn.ModuleList(fully_connected_layers)

        # raise NotImplementedError()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.08, 0.08)
        # raise NotImplementedError()

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embed.weight = torch.nn.Parameter(torch.from_numpy(emb))
        self.embed.weight.requires_grad = True
        # raise NotImplementedError()

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """

        X = self.embed(x)
        X = X.mean(dim=1)

        # X = self.emb_drop(X)
        # X = self.emb_bn(X)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                X = F.relu(layer(X))
            else:
                X = layer(X)
        return X

        # raise NotImplementedError()
