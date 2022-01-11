import sys
import csv

def find_idx(lst, elem):
    idx_lst = []
    idx = 0
    while (idx <= (len(lst) - 1)):
        if (elem in lst[idx:]):
            idx = lst.index(elem, idx)
            idx_lst.append(idx)
            idx += 1
        else:
            break
    return idx_lst

def result_catagory(result_training):
    result = []
    while len(result) < 2:
        for i in range(len(result_training)):
            if result_training[i] not in result:
                result.append(result_training[i])
    return result

def decision(attr_training, result_training):
    reseult_catagory = result_catagory(result_training)
    attr_catagory = result_catagory(attr_training)
    attr_y = attr_catagory[0]
    attr_n = attr_catagory[1]
    result_y = reseult_catagory[0]
    result_n = reseult_catagory[1]

    result_y_idx = find_idx(result_training, result_y)
    result_n_idx = find_idx(result_training, result_n)
    count_attrY_resultY = 0
    count_attrY_resultN = 0
    count_attrN_resultY = 0
    count_attrN_resultN = 0

    Y_result = ''
    N_result = ''
    for i in range(len(attr_training)):
        if attr_training[i] == attr_y:
            if i in result_y_idx:
                count_attrY_resultY += 1
            else:
                count_attrY_resultN += 1
        else:
            if i in result_y_idx:
                count_attrN_resultY += 1
            else: 
                count_attrN_resultN += 1
    
    if count_attrY_resultY > count_attrY_resultN:
        #yes in attr goes to yes
        Y_result = reseult_catagory[0]
    else:
        Y_result = reseult_catagory[1]
    if count_attrN_resultY > count_attrN_resultN:
        #no in attr goes to yes
        N_result = reseult_catagory[0]
    else:
        N_result = reseult_catagory[1]
    return attr_y, attr_n, Y_result, N_result
    
def decision_stump(attr_training, result_training):
    attr_y, attr_n, Y_result, N_result = decision(attr_training, result_training)
    result = []
    for i in range(len(attr_training)):
        if attr_training[i] == attr_y:
            result.append(Y_result)
        else:
            result.append(N_result)
    return result

def error(result, predict):
    error = 0
    for i in range(len(result)):
        if result[i] != predict[i]:
            error += 1
    return error / len(result)



if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_idx = sys.argv[3]
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_out = sys.argv[6]
    training_attr = []
    training_result = []
    test_attr = []
    test_result = []
    with open(train_input, newline = '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            training_attr.append(row[0].split()[int(split_idx)])
            training_result.append(row[0].split()[-1])
    training_out = decision_stump(training_attr[1:], training_result[1:])
    with open(test_input, newline = '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            test_attr.append(row[0].split()[int(split_idx)])
            test_result.append(row[0].split()[-1])
    testing_out = decision_stump(test_attr[1:], test_result[1:])

    textfile1 = open(train_output, "w")
    for elem in training_out:
        textfile1.write(elem + "\n")
    textfile1.close()

    textfile2 = open(test_output, "w")
    for elem in testing_out:
        textfile2.write(elem + "\n")
    textfile2.close()

    error_train = error(training_result[1:], training_out)
    error_training = 'error(train): '+ str(error_train)

    error_test = error(test_result[1:], testing_out)
    error_testing = 'error(test): '+ str(error_test)
    
    metrics = [error_training, error_testing]
    textfile3 = open(metrics_out, "w")
    for elem in metrics:
        textfile3.write(elem + "\n")
    textfile3.close()

