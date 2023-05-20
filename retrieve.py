#!/usr/bin/env python3
import math
from utils import keyword_weights
from tqdm import tqdm


# open paragraph file
file = open('data/paragraphs.txt', 'r', encoding='UTF-8')
para = file.readlines()
avg_para_len = len("".join(para)) / len(para)


def count(sentence, target):
    return sentence.count(target)


def idf(target_word):
    cnt = 0
    for sentences in para:
        if target_word in sentences:
            cnt += 1
    return math.log((len(para) - cnt + 0.5) / (0.5 + cnt) + 1)


def score(sentence, target, const_k, const_b):
    return (const_k + 1) * count(sentence, target) / (
            1 - const_b + const_b * len(sentence) / avg_para_len + count(sentence, target))


def retrieve(
        query,
        return_list_len=1,
        const_k=1.5,
        const_b=0.75,
        min_keyword_weight=0.43,
):
    # get weights of each word in query
    q_word_weight = keyword_weights(query)
    # filtering keywords
    q_word_weight = {k: v for k, v in
                     filter(lambda x: x[1] > min_keyword_weight, q_word_weight.items())}

    rsv_list = []
    # evaluate through each paragraph
    for para_index in tqdm(range(len(para))):

        paragraph = para[para_index]
        rsv = 0
        words_contribute = []

        for word in q_word_weight.keys():
            # calculate rsv
            words_contribute.append(
                (idf(word) * score(paragraph, word, const_k, const_b), word)
            )
            rsv += words_contribute[-1][0]

        rsv_list.append({
            'index': para_index,
            'paragraph': para[para_index],
            'score': rsv,
            'word_scores': words_contribute
        })

    rsv_list = sorted(rsv_list, key=lambda x: x['score'], reverse=True)

    # normalize the score
    score_sum = sum(x['score'] for x in rsv_list)
    for i in range(len(rsv_list)):
        rsv_list[i]['score'] /= score_sum + 1e-9

    return rsv_list[0:min(len(rsv_list), return_list_len)]
