import string
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def get_vocab():
    f = open('data/rt-polaritydata/rt-polarity.neg', "rb")
    neg_lines = f.read()
    f.close()

    f = open('data/rt-polaritydata/rt-polarity.pos', "rb")
    pos_lines = f.read()
    f.close()

    f = open('data/vocabulary.txt', "r")
    w2v_vocab = f.read()
    f.close()

    w2v_vocab = set(w2v_vocab.split())
    # print(embed_vocab)

    lines = neg_lines+pos_lines
    lines = lines.decode('utf-8', 'ignore')

    # lines = lines.translate(str.maketrans('', '', string.punctuation))
    lines = clean_str(lines)
    data_vocab = set(lines.split())
    # print(vocab)

    dif = data_vocab.difference(w2v_vocab)
    same = data_vocab.intersection(w2v_vocab)
    print(len(data_vocab), len(same), len(dif))
    #print(same)
    return same, dif


def get_data():
    f = open('data/rt-polaritydata/rt-polarity.neg', "rb")
    neg_lines = f.read().decode('utf-8', 'ignore')
    f.close()

    f = open('data/rt-polaritydata/rt-polarity.pos', "rb")
    pos_lines = f.read().decode('utf-8', 'ignore')
    f.close()

    neg_lines = neg_lines.split('\n')
    for i in range(len(neg_lines)):
        neg_lines[i] = clean_str(neg_lines[i])

    pos_lines = pos_lines.split('\n')
    for i in range(len(pos_lines)):
        pos_lines[i] = clean_str(pos_lines[i])

    return pos_lines, neg_lines
