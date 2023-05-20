#!/usr/bin/env python3
import utils
from retrieve import retrieve
from answering import question_answering
from page_search import find_page


answer_conf_threshold = 0.28

# input
question = input("请输入问题\n")

# retrieve relevant paragraphs
paragraphs = retrieve(question, return_list_len=5)

# reading comprehension
print("正在筛选")
answers = []
for paragraph in paragraphs:
    print(paragraph['score'], paragraph['paragraph'])
    answer = question_answering(question, paragraph['paragraph'])
    answer['paragraph'] = paragraph['paragraph']
    answer['para_score'] = paragraph['score']
    answer['para_loc'] = find_page(paragraph['paragraph'])
    answer['similarity'] = utils.similarity(question, answer['answer'])
    answer['total'] = (answer['similarity'] + answer['para_score'] + answer['conf']) / 3
    answers.append(answer)

# filter conf
answers = list(filter(lambda x: x['total'] > answer_conf_threshold, answers))

# sort by similarity of answer and paragraph
answers = sorted(answers, key=lambda x: x['total'], reverse=True)
answer_texts = [
    {'answer': x['answer'],
     'conf': x['total'],
     'location': x['para_loc'],
     'source': x['paragraph']
     }
    for x in answers
]

print("结果：")
if not answer_texts:
    print("无答案")

print(*answer_texts, sep='\n')
