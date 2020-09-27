import os
import sys
import torch
import numpy as np
from transformers import *

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset_env'))

import file_storage

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

def get_similar_traj(config, instruction):
    lang_model = LanguageModelInstructionEncoder(config.lang_model)
    cos_sim = torch.nn.CosineSimilarity()
    instruction_words = lang_model(instruction)

    trajectories = []
    all_instructions = file_storage.get_instruct_traj()

    for i in range(instruction_words.shape[0]):
        word = instruction_words[i]
        max_index, max_sim = 0, -10
        for j, (search_instruction, trajectory) in enumerate(all_instructions):
            search_words = lang_model(search_instruction)
            for k in range(search_words.shape[0]):
                word2 = search_words[k]
                word, word2 = torch.reshape(word, (1, -1)), torch.reshape(word2, (1, -1))
                sim = cos_sim(word, word2)

                if sim > max_sim:
                    max_sim = sim
                    max_index = j

        trajectories.append(all_instructions[j][1])
    return trajectories

if __name__ == '__main__':
    print("=> Testing common.py")
    lang_model = LanguageModelInstructionEncoder('bert')
    out = lang_model("hello world suck add token")
    print(out['word_embeddings'].shape)