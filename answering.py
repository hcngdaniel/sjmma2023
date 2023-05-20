#!/usr/bin/env python3
import utils
import torch
from transformers import BertForQuestionAnswering


device = 'cuda'
qa_model = BertForQuestionAnswering.from_pretrained('pretrained/roberta-base-chinese-extractive-qa')
qa_model.to(device)


def question_answering(question, context):
    token = utils.tokenize_for_qa(question, context).to(device)
    out = qa_model(
        **token
    )
    start_prob = torch.softmax(out.start_logits.detach().cpu(), dim=1)
    end_prob = torch.softmax(out.end_logits.detach().cpu(), dim=1)
    start_prob[0][0] = 0.
    end_prob[0][0] = 0.
    start_prob *= token['token_type_ids'].cpu()
    end_prob *= token['token_type_ids'].cpu()
    start_idx = torch.argmax(start_prob)
    end_idx = torch.argmax(end_prob)
    answer = utils.tokenizer.decode(token['input_ids'][0][start_idx:end_idx + 1]).replace(" ", "")
    return {
        'start': start_idx,
        'end': end_idx,
        'answer': answer,
        'conf': (start_prob[0][start_idx] + end_prob[0][end_idx]) / 2
    }
