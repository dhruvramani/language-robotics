import re
import os 
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchtext
from torchtext.data import get_tokenizer

from transformers import *

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from layers import GaussianNetwork
from dataset_env import file_storage

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

MODELS = {"bert":     (BertModel,       BertTokenizer,       'bert-base-uncased',     768),
          "gpt":      (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt',            768),
          "gpt_2":    (GPT2Model,       GPT2Tokenizer,       'gpt2',                  768),
          "ctrl":     (CTRLModel,       CTRLTokenizer,       'ctrl',                  1280),
          "tr_xl":    (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103',      1024),
          "xlnet":    (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased',      768),
          "xlm":      (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024',     1024),
          "dis_bert": (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased', 768),
          "roberta":  (RobertaModel,    RobertaTokenizer,    'roberta-base',          768),
          "xlm_rob":  (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base',      768),
          "muse":     (hub.load, None ,'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3', 512)
         }

class LanguageModelInstructionEncoder(torch.nn.Module):
    def __init__(self, lang_model_key, latent_dim=32):
        super(LanguageModelInstructionEncoder, self).__init__()
        self.lang_model_key = lang_model_key.lower()
        self.latent_dim = latent_dim
        assert self.lang_model_key in MODELS.keys()
        
        self.lang_model_config = MODELS[self.lang_model_key]
        if self.lang_model_key == "muse":
            self.lang_model = self.lang_model_config[0](self.lang_model_config[2])
            self.lang_callback = self.callback_muse
        else:
            self.tokenizer = self.lang_model_config[1].from_pretrained(self.lang_model_config[2])
            self.lang_model = self.lang_model_config[0].from_pretrained(self.lang_model_config[2])
            self.lang_callback = self.callback_huggingface

        self.goal_dist = GaussianNetwork(self.lang_model_config[-1], self.latent_dim)

    def callback_muse(self, text):
        assert self.lang_model_key == "muse"
        with torch.no_grad():
            embedding = torch.from_numpy(self.lang_model(text).numpy())
        return embedding.float().to(device)

    def callback_huggingface(self, text):
        assert self.lang_model_key != "muse"
        input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            # Source : https://github.com/huggingface/transformers/issues/1950#issuecomment-558770861
            # So instead of having to average of all tokens and use that as a sentence representation,
            # it is recommended to just take the output of the [CLS] which then represents the whole sentence.
            out = self.lang_model(input_ids)
            cls_embedding = out[0][0]
        return cls_embedding[0].reshape(1, -1) # NOTE: [0].reshape(1, -1) might be incorrect - see docs.

    def forward(self, text):
        embedding = self.lang_callback(text)
        z, dist, mean, std = self.goal_dist(embedding)
        return z, dist, mean, std
        

class BasicInstructionEncoder(torch.nn.Module):
    def __init__(self, latent_dim=32, embedding_dim=8, arch='AvgPool'):
        '''
            - NOTE : Mostly not gonna be using this, hence might have errors. 
            + A simple mapping from sentence `s` describing the goal to latent embedding z.
            + Arguments
                - latent_dim: the size of the latent embedding, pass from config.
                - arch: choices=['RNN', 'LSTM', 'AvgPool']
        '''
        super(BasicInstructionEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.arch_type = arch.upper()
        assert self.arch_type in ['RNN', 'LSTM', 'AVGPOOL']

        self.tokenizer = get_tokenizer("basic_english")
        self.word2index = file_storage.get_vocab2idx()
        self.embedding = torch.nn.Embedding(len(self.word2index), self.embedding_dim)

        if self.arch_type == 'RNN':
            self.encoder = torch.nn.RNN(self.embedding_dim, self.embedding_dim / 2, num_layers=1, bidirectional=False, batch_first=True)
        elif self.arch_type == 'LSTM':
            self.encoder = torch.nn.RNN(self.embedding_dim, self.embedding_dim / 2, num_layers=1, bidirectional=False, batch_first=True)
        elif self.arch_type == 'AVGPOOL':
            self.encoder = torch.nn.AvgPool1d(self.embedding_dim / 4)

        self.goal_dist = GaussianNetwork(self.embedding_dim / 2, self.latent_dim)

    def forward(self, text):
        # TODO *IMPORTANT* : Deal with multiple batch_sizes
        text = re.sub("[^\w]", " ",  text)
        tokens = self.tokenizer(text)
        idxs = torch.LongTensor([[self.word2index[word] for word in tokens]])
        embedding = self.embedding(idxs)
        embedding = torch.reshape(embedding, (1, 1, -1))
        output = self.encoder(embedding)
        
        z, mean, std = self.goal_dist(output)

        return z, mean, std