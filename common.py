import os
import sys
import torch
import numpy as np
from transformers import *

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
         }

class LanguageModelInstructionEncoder(torch.nn.Module):
    def __init__(self, lang_model_key):
        super(LanguageModelInstructionEncoder, self).__init__()
        self.lang_model_key = lang_model_key.lower()
        assert self.lang_model_key in MODELS.keys()
        
        self.lang_model_config = MODELS[self.lang_model_key]

        self.tokenizer = self.lang_model_config[1].from_pretrained(self.lang_model_config[2])
        self.lang_model = self.lang_model_config[0].from_pretrained(self.lang_model_config[2])

    def lang_callback(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            out = self.lang_model(input_ids)
            word_embeddings = out[0][0][1:]
            sentence_embedding = out[0][0][0].reshape(1, -1)
        return {'word_embeddings' : word_embeddings, 'sentence_embedding' : sentence_embedding}

    def forward(self, text):
        embedding = self.lang_callback(text)
        return embedding

if __name__ == '__main__':
    print("=> Testing common.py")
    lang_model = LanguageModelInstructionEncoder('bert')
    out = lang_model("hello world suck add token")
    print(out['word_embeddings'].shape)