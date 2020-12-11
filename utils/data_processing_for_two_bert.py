import numpy as np
import pandas as pd
import json

import requests
import urllib

from transformers import BertTokenizer, BertForQuestionAnswering, BertTokenizerFast
import torch

import data_processing

MAX_LENGTH = 512  #max input length that bert model can accept

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        if answers[i]['text'] == 'no-answer':
            print("no-answer!")
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(encodings.char_to_token(i,answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i,answers[i]['answer_end']-1))
            #if none, the answer span has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions':start_positions,'end_positions':end_positions})    

def data_processing_for_two_bert(url):
    response = urllib.request.urlopen(url)
    raw = pd.read_json(response)
    contexts, questions, answers, ids = data_processing.load_data(raw)
    data_processing.add_end_idx(answers,contexts)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    context_encodings = tokenizer(contexts, truncation = True, padding = True)
    question_encodings = tokenizer(questions, truncation = True, padding = True)
    add_token_positions(context_encodings, answers, tokenizer)

    encodings = {}
    encodings['context_input_ids'] = context_encodings['input_ids']
    encodings['context_attention_mask'] = context_encodings['attention_mask']
    encodings['question_input_ids'] = question_encodings['input_ids']
    encodings['question_attention_mask'] = question_encodings['attention_mask']
    encodings['start_positions'] = context_encodings['start_positions']
    encodings['end_positions'] = context_encodings['end_positions']

    return encodings, answers

if __name__ == "__main__":
    encodings, answers =  data_processing_for_two_bert("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")

    print("length of encoding",len(encodings['context_input_ids']))