import sys
import csv
import numpy as np
csv.field_size_limit(sys.maxsize)


#python3 forwardbackward.py toy_data/validation.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_output/hmminit.txt toy_output/hmmemit.txt toy_output/hmmtrans.txt 1_predicted.txt 1_metrics.txt
#python3 forwardbackward.py en_data/validation.txt en_data/index_to_word.txt en_data/index_to_tag.txt en_output/hmminit.txt en_output/hmmemit.txt en_output/hmmtrans.txt 2_predicted.txt 2_metrics.txt
#python3 forwardbackward.py fr_data/validation.txt fr_data/index_to_word.txt fr_data/index_to_tag.txt fr_output/hmminit.txt fr_output/hmmemit.txt fr_output/hmmtrans.txt 3_predicted.txt 3_metrics.txt

def logsumexp(l):
    m = np.max(l)
    return m + np.log(np.sum(np.exp(l - m)))


def forwardbackward(words, word2idx, idx2word, tag2idx, idx2tag, pai, B, A):
    word_only = []
    label = []
    #init alpha
    alpha = np.zeros((len(words),len(tag2idx)))
    #get alpha_1
    word0, tag = words[0][0].split("\t")
    label.append(tag)
    word_only.append(word0)
    alpha[0]=np.log(A[:,word2idx[word0]])+np.log(pai)
    #get the rest of alpha entries
    for i in range(1,len(words)):
        if words[i][0] != ['']:
            word, tag = words[i][0].split("\t")
            label.append(tag)
            word_only.append(word)
            
            for j in range(len(tag2idx)):
                print(logsumexp(alpha[i-1] + np.log(B[:,j])))
                alpha[i][j] =np.log(A[j][word2idx[word]])+logsumexp(alpha[i-1] + np.log(B[:,j]))

                '''for k in range(len(tag2idx)):
                    total += np.exp(alpha[i-1][k]+np.log(B[k][j]))
                alpha[i][j] = np.log(A[j][word2idx[word]]) + np.log(total)'''
    loglikelihood = logsumexp(alpha[-1])
    
    #init beta 
    beta = np.zeros((len(words), len(tag2idx)))
    #get beta_T
    beta[-1] = np.log(np.ones((1, len(tag2idx))))
    #print(beta)
    for i in range(2,len(words)+1):
        word, tag = words[-i+1][0].split("\t")
        for j in range(len(tag2idx)):
            #print("------",np.log(A[:,word2idx[word]])+beta[-i+1] + np.log(B[j]))
            beta[-i][j] = logsumexp(np.log(A[:,word2idx[word]])+beta[-i+1] + np.log(B[j]))

            '''for k in range(len(tag2idx)):
                total += np.exp(beta[-i+1][k]+np.log(B[k][j])+np.log(A[j][word2idx[word]]))

            beta[-i][j] = np.log(total)'''

    predict_idx = np.argmax(alpha+beta, axis = 1)
    predict = np.vectorize(idx2tag.get)(predict_idx,)
    return predict, loglikelihood, word_only, label

def acc(predict, label):
    count = 0
    for i in range(len(predict)):
        if predict[i]==label[i]:
            count += 1
    return count

if __name__ == '__main__':
    validation_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    with open(validation_input) as f:
	    val_in = [item.replace("\n", "").split(" ") for item in f.readlines()]
    index2word = np.loadtxt(index_to_word, dtype=str)
    index2tag = np.loadtxt(index_to_tag, dtype=str, delimiter = " ")
    i2w = list(range(len(index2word)))
    i2t = list(range(len(index2tag)))
    word2idx = dict(zip(index2word, i2w))
    idx2word = dict(zip(i2w, index2word))
    tag2idx = dict(zip(index2tag,i2t))
    idx2tag = dict(zip(i2t, index2tag))
    pai = np.loadtxt(hmminit)
    B = np.loadtxt(hmmtrans)
    A = np.loadtxt(hmmemit)
    formatted_val = [[] for _ in range(val_in.count([""])+1)]
    count = 0
    for i in range(len(val_in)):
        if val_in[i] == [""]:
            count += 1
        else:
            formatted_val[count].append(val_in[i])
    

    total_log = 0
    acc_total = 0
    total = 0
    out_w = []
    out_p = []
    for line in formatted_val:
        predict, loglikelihood,word_only,label= forwardbackward(line, word2idx, idx2word, tag2idx, idx2tag, pai, B, A)
        total_log += loglikelihood
        acc_total += acc(predict, label)
        total += len(predict)
        out_w.append(word_only)
        out_p.append(predict)

    textfile1 = open(predicted_file, "w")
    for i in range(len(out_w)):
        for j in range(len(out_w[i])):
            textfile1.write(out_w[i][j] + "\t" + out_p[i][j] + "\n")
        textfile1.write("\n")
    textfile1.close()
    #print(predict, loglikelihood,word_only,label)

    


    textfile2 = open(metric_file, "w")
    textfile2.write("Average Log-Likelihood: %f\n" %(total_log/len(formatted_val)))
    textfile2.write("Accuracy: %f\n" %(acc_total/total))
    textfile2.close()