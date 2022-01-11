import sys
import csv
import numpy as np
csv.field_size_limit(sys.maxsize)


#python3 learnhmm.py toy_data/train.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt 1_hmminit.txt 1_hmmemit.txt 1_hmmtrans.txt
#python3 learnhmm.py en_data/train.txt en_data/index_to_word.txt en_data/index_to_tag.txt 2_hmminit.txt 2_hmmemit.txt 2_hmmtrans.txt

def learning(formatted_train, word2index, index2word, tag2index, index2tag):
    pai = np.zeros(len(tag2index))
    B = np.zeros((len(tag2index), len(tag2index)))
    A = np.zeros((len(tag2index), len(word2index)))
    for line in formatted_train:
        for i in range(len(line)):
            word = line[i]
            word, tag = word[0].split("\t")
            word_idx, tag_idx = word2index[word], tag2index[tag]
            if i==0:
                pai[tag_idx] += 1
            if i < len(line)-1:
                next = line[i+1]
                next, nexttag = next[0].split("\t")
                nextword_idx, nexttag_idx = word2index[next], tag2index[nexttag]
                B[tag_idx][nexttag_idx] += 1
            A[tag_idx][word_idx] += 1

    pai += 1
    B += 1
    A += 1
    pai /= np.sum(pai)
    B = B.reshape(len(tag2index), -1)
    B /= np.sum(B, axis = 1, keepdims = True)
    A /= np.sum(A, axis = 1).reshape(len(tag2index), -1)
    return pai, np.transpose(B), A

 
if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    with open(train_input) as f:
	    train_in = [item.replace("\n", "").split(" ") for item in f.readlines()]
    formatted_train = [[] for _ in range(train_in.count([""])+1)]
    count = 0
    for i in range(len(train_in)):
        if train_in[i] == [""]:
            count += 1
        else:
            formatted_train[count].append(train_in[i])

    index2word = np.loadtxt(index_to_word, dtype=str)
    index2tag = np.loadtxt(index_to_tag, dtype=str, delimiter = " ")
    i2w = list(range(len(index2word)))
    i2t = list(range(len(index2tag)))
    word2idx = dict(zip(index2word, i2w))
    idx2word = dict(zip(i2w, index2word))
    tag2idx = dict(zip(index2tag,i2t))
    idx2tag = dict(zip(i2t, index2tag))
    pai, B, A = learning(formatted_train[:10000], word2idx, idx2word, tag2idx, idx2tag)
    pai = [str(x) for x in pai]
    textfile1 = open(hmminit, "w")
    for elem in pai:
        textfile1.write(elem + "\n")
    textfile1.close()

    textfile2 = open(hmmemit, "w")
    for elem in A:
        elem = [str(x) for x in elem]
        textfile2.write(" ".join(elem) + "\n")
    textfile2.close()

    textfile3 = open(hmmtrans, "w")
    for elem in np.transpose(B):
        elem = [str(x) for x in elem]
        textfile3.write(" ".join(elem) + "\n")
    textfile3.close()
    

