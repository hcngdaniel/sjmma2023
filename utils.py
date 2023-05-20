#!/usr/bin/env python3
import torch
import jieba
from transformers import BertTokenizer, BertModel


device = 'cuda'
tokenizer = BertTokenizer.from_pretrained('pretrained/bert-base-chinese')
bert = BertModel.from_pretrained('pretrained/bert-base-chinese')
bert.to(device)


def tokenize(text):
    return tokenizer.encode_plus(
        text=text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding='max_length',
    )


def tokenize_for_qa(question, context):
    return tokenizer.encode_plus(
        text=question,
        text_pair=context,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding='max_length',
    )


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def keyword_weights(text, min_weight=0.1):

    # calc the embedding of the text
    # tokenize
    text_token = tokenize(text).to(device)
    # forward
    text_cls = bert(
        **text_token
    )
    # pooling
    text_cls = mean_pooling(text_cls, text_token['attention_mask'])
    # post-process
    text_cls = text_cls.detach().cpu()

    # split the text
    words = list(jieba.cut(text))

    similarities = []

    # calc the sim with the text for each word
    for word in words:
        # tokenize
        word_token = tokenize(word).to(device)
        # forward
        word_cls = bert(
            **word_token
        )
        # pool
        word_cls = mean_pooling(word_cls, word_token['attention_mask'])
        # post-process
        word_cls = word_cls.detach().cpu()
        # calculate sim
        sim = torch.cosine_similarity(text_cls, word_cls)
        similarities.append(sim)

    # stacking
    similarities = torch.stack(similarities).view(-1)

    # normalize
    similarities -= torch.min(similarities)
    similarities = similarities / torch.max(similarities) * (1 - min_weight) + min_weight

    # sort
    sorted_sim = torch.topk(similarities, len(words))
    keywords = {words[index]: score for index, score in zip(sorted_sim.indices, sorted_sim.values)}

    return keywords


def similarity(question, answer):
    question_token = tokenize(question).to(device)
    answer_token = tokenize(answer).to(device)
    question_cls = bert(
        **question_token
    )
    question_cls = mean_pooling(question_cls, question_token['attention_mask'])
    answer_cls = bert(
        **answer_token
    )
    answer_cls = mean_pooling(answer_cls, answer_token['attention_mask'])
    sim = torch.cosine_similarity(question_cls, answer_cls).detach().cpu()
    return sim


