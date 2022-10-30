# Defined in Section 2.2.2
# 这也是一个分句的思路
from os import name
from typing import Mapping


def load_dict():
    f = open("lexicon.txt")
    lexicon = set()
    max_len = 0
    for line in f:
        word = line.strip()
        lexicon.add(word)
        if len(word) > max_len:
            max_len = len(word)
    f.close()

    return lexicon, max_len

def fmm_word_seg(sentence, lexicon, max_len):
    begin = 0
    end = min(begin + max_len, len(sentence))
    words = []
    while begin < end:
        word = sentence[begin:end]
        if word in lexicon or end - begin == 1:
            words.append(word)
            begin = end
            end = min(begin + max_len, len(sentence))
        else:
            end -= 1
    return words

if __name__ == '__main__':
    lexicon, max_len = load_dict()
    words = fmm_word_seg(input("请输入句子："), lexicon, max_len)

    for word in words:
        print(word,) 
'''
请输入句子：我爱你，你喜欢我 吗              
我
爱
你
，
你
喜欢
我
吗
'''        
