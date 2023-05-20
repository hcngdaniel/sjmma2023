#!/usr/bin/env python3
from tqdm import tqdm
import retrieve


u_data = open("data/7up.txt", "r", encoding='UTF-8').read()  # data of 7up
d_data = open("data/7down.txt", "r", encoding='UTF-8').read()  # data of 7down


def rm(x):
    return x.replace("\n", "")


def find_page(sentence):
    first_half = sentence[0:len(sentence) // 2]
    second_half = sentence[len(sentence) // 2:len(sentence)]
    wb = "上册"
    context = u_data
    if first_half in d_data or second_half in d_data:
        context = d_data
        wb = "下册"
    if first_half in u_data or second_half in u_data:
        context = u_data
        wb = "上册"
    context = context.replace(second_half, "|  |").replace(first_half, "|  |").split("|  |")
    sp = context[len(context) - 1].split("\n")
    el = 0
    while el < len(sp):
        if sp[el][0:4] != "\\end":
            el += 1
        else:
            el = int(sp[el][5:len(sp[el]) - 1])
            break
    fp = context[0].split("\n")
    nl = len(fp) - 1
    while nl >= 0:
        if fp[nl][0:6] != "\\begin":
            nl -= 1
        else:
            nl = int(fp[nl][7:len(fp[nl]) - 1])
            break
    return {'book': wb, 'start': nl, 'end': el}
