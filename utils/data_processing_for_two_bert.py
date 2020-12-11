import numpy as np
import pandas as pd
import json

import requests
import urllib

from transformers import BertTokenizer, BertForQuestionAnswering, BertTokenizerFast
import torch

MAX_LENGTH = 512  #max input length that bert model can accept

def load_data(train_df):
    contexts = []
    questions = []
    answers = []
    ids = []
    for i in range(train_df['data'].shape[0]):
        topic = train_df['data'].iloc[i]['paragraphs']
        for sub_para in topic:
            context = sub_para['context']
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                contexts.append(context)
                ids.append(q_a['id'])
                if q_a['is_impossible'] is False:
                    answers.append(q_a['answers'][0])
                else:
                    answer = {}
                    answer['answer_start'] = 0
                    answer['text'] = 'no-answer'
                    answers.append(answer)

    return contexts, questions, answers, ids

def add_end_idx(answers, contexts):
    for answer, context in zip(answers,contexts):
        if answer['text'] == "no-answer":
            answer['answer_end'] = 0
        else:
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)
            #sometimes SQuAD answers are off by a character or two 
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1
            elif context[start_idx-2: end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2

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
    contexts, questions, answers, ids = load_data(raw)
    add_end_idx(answers,contexts)
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