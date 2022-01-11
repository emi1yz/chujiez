import sys
import csv
import numpy as np
csv.field_size_limit(sys.maxsize)

#python3 neuralnet.py tinyTrain.csv tinyValidation.csv tinyTrain_out.labels tinyValidation_out.labels tinyMetrics_out.txt 1 4 2 0.1

def linearForward(x, para):
    return np.dot(x, np.transpose(para))

def sigmoidForward(a):
    return np.exp(a)/(1 + np.exp(a))

def softmaxForward(b):
    return np.exp(b)/np.sum(np.exp(b))

def crossEntropyForward(true_y, y_hat):
    if len(y_hat.shape) == 1:
        N=1
        true_y = int(true_y)
        K = y_hat.shape[0]
        y = np.zeros((K))
        total = 0
        h_y = int(true_y)
        y[h_y] = 1
        for k in range(K):
            total += y[k] * np.log(y_hat[k])
        total = - total 
        return total
'''    else:
        true_y = true_y.astype(int)
        N, K = y_hat.shape
        y = np.zeros((N,K))
        total = 0
        for i in range(N):
            h_y = true_y[i]
            y[i][h_y] = 1
            for k in range(K):
                #print(y[i][k] * np.log(y_hat[i][k]))
                total += y[i][k] * np.log(y_hat[i][k])
        return - total / N'''

def mean_cross_entropy(x,y,alpha,beta):
    #print(x)
    total = 0
    for i in range(x.shape[0]):
        #print("xi+++++",x[i])
        x1, a, z, b, y_hat, J = NNFORWARD(x[i], y[i], alpha, beta)
        total += J
    return total/x.shape[0]

def NNFORWARD(x, y, alpha, beta):
    #print("x-------", x)
    #1x6
    M = np.shape(x)
    #1,4
    a = linearForward(x, alpha)
    #1,4
    z = sigmoidForward(a)
    if len(x.shape) == 1:
        ones = np.ones((1))
    else:
        ones = np.ones((len(x),1))
    #1,5
    z = np.hstack((ones,z))
    #1,4
    b = linearForward(z, beta)
    #1,4
    y_hat = softmaxForward(b)
    #1
    J = crossEntropyForward(y, y_hat)
    #print("Cross entropy", J)
    return x, a, z, b, y_hat, J

def softmaxBackward(hot_y, y_hat):
    K = y_hat.shape[0]
    res= np.zeros((K))
    for j in range(K):
        hot_y_cur = hot_y
        if (j!=hot_y_cur):
            res[j] = y_hat[j]
        else:
            res[j] = -1 + y_hat[j]
    return res

def linearBackward(prev, p, grad_curr):
    grad_param=np.array(np.outer(grad_curr,prev))
    grad_prevl = np.dot(grad_curr, p)
    return grad_param, grad_prevl


def sigmoidBackward(curr, grad_curr):
    grad_prevl = np.multiply(np.multiply(curr, (1 - curr)), grad_curr)
    return grad_prevl


def NNBackward(x, y, alpha, beta, z, y_hat):
    g_b = softmaxBackward(y, y_hat)
    #print("g_b",g_b)
    g_beta, g_z = linearBackward(z, beta, g_b)
    #print("g_beta",g_beta)
    
    g_z = np.delete(g_z,0)
    z = np.delete(z,0)
    #print("g_z",g_z)
    g_a = sigmoidBackward(z, g_z)
    #print("g_a",g_a)
    g_alpha, g_x = linearBackward(x, alpha, g_a)
    #print("g_alpha",g_alpha)
    return g_alpha, g_beta, g_b, g_z, g_a


def adagrad(st, gradient, p, learning_rate):
    st += (gradient * gradient)
    p = p - learning_rate/(np.sqrt(st+0.00001))*gradient
    return st, p



    





'''x1 = np.array([[1,0,1,1,1,0],[1,0,1,1,1,0]])
x2= np.array([[1,0,0,0,1,0]])
y1 = np.array([[2]])
y2 = np.array([[3]])
N,M = x1.shape

K=4
D=4
alpha = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
beta = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
x, a, z, b, y_hat, J = NNFORWARD(x1,y1,alpha,beta)'''
'''print("x: ",x)
print("a: ",a)
print("z: ",z)
print("b: ",b)
print("yhat: ", y_hat)
print("J: ",J)'''
'''g_alpha, g_beta, g_b, g_z, g_a = NNBackward(x1, y1, alpha, beta, z, y_hat)
print("g_b",g_b)
print("g_beta",g_beta)
print("g_z", g_z)
print("g_a", g_a)
print("g_alpha",g_alpha)
st = np.zeros((4,6))
new_alpha = adagrad(st, g_alpha, alpha, 0.1)
#print(new_alpha)
st1 = np.zeros((4,5))
new_beta = adagrad(st1, g_beta, beta, 0.1)
print(new_beta)'''

def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    N_train, M = tr_x.shape
    N_valid, M = valid_x.shape
    if (init_flag=='2'):
        alpha = np.zeros((hidden_units, M-1))
        beta = np.zeros((4, hidden_units))
    else:
        alpha = np.random.uniform(low=0, high=0.2, size=(hidden_units, M-1) )-0.1
        beta = np.random.uniform(low=0, high=0.2, size=(4, hidden_units) )-0.1
    a_bias = np.zeros((hidden_units,1))
    b_bias = np.zeros((4 ,1))

    alpha = np.hstack((a_bias,alpha))
    beta = np.hstack((b_bias,beta))
    
    N_a,M_a = alpha.shape
    N_b,M_b = beta.shape
    train_entropy = np.zeros((num_epoch))
    valid_entropy = np.zeros((num_epoch))
    
    st_a = np.zeros((N_a,M_a))
    st_b = np.zeros((N_b,M_b))
    for i in range(num_epoch): 
        #print("alpha===", alpha)
        for j in range(N_train):
            x, a, z, b, y_hat, J = NNFORWARD(tr_x[j], tr_y[j], alpha, beta)
            print("++++++++++++++++")
            print(a, z, b, y_hat, J)
            g_alpha, g_beta, g_b, g_z, g_a = NNBackward(tr_x[j], tr_y[j], alpha, beta, z, y_hat)
            print("---------------------")
            print(g_alpha, g_beta, g_b, g_z, g_a)
            st_a, alpha = adagrad(st_a, g_alpha, alpha, learning_rate)
            st_b, beta = adagrad(st_b, g_beta, beta, learning_rate)
            print("alpha+++", alpha)
            print("beta+++", beta)
        J_train = mean_cross_entropy(tr_x,tr_y,alpha,beta)
        train_entropy[i] = J_train
        J_v = mean_cross_entropy(valid_x,valid_y,alpha,beta)
        valid_entropy[i] = J_v
        
    return alpha, beta, train_entropy, valid_entropy


def predict(alpha, beta, x_data):
    predicted_label = np.zeros(x_data.shape[0])
    for i in range(x_data.shape[0]):
        #print("x_data[i]",x_data[i])
        a = linearForward(x_data[i], alpha)
        z = sigmoidForward(a)
        ones = np.ones((1))
        z = np.hstack((ones,z))
        b = linearForward(z, beta)
        y_hat = softmaxForward(b)
        label = np.argmax(y_hat)
        predicted_label[i] = label
    return predicted_label


def error(train_predict, y_train, val_predict, y_val):
    count_train = 0
    count_test = 0
    for i in range(len(y_train)):
        if y_train[i] != train_predict[i]:
            count_train += 1
    for j in range(len(y_val)):
        if y_val[j] != val_predict[j]:
            count_test += 1
    return count_train/len(y_train), count_test/len(y_val)

if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = sys.argv[8]
    learning_rate = float(sys.argv[9])


    train_in = np.loadtxt(sys.argv[1], delimiter=',')
    valid_in = np.loadtxt(sys.argv[2], delimiter=',')

    X_train = train_in[:, 1:]
    y_train = train_in[:, 0]
    X_val = valid_in[:, 1:]
    y_val = valid_in[:, 0]

    N_train, M = X_train.shape
    N_valid, M = X_val.shape

    x_bias = np.ones((N_train,1))
    x_v_bias = np.ones((N_valid,1))
    X_train = np.hstack((x_bias,X_train))
    X_val = np.hstack((x_v_bias,X_val))


    '''print("X_train",X_train)
    print("y_train",y_train)
    print("X_val",X_val)
    print("y_val",y_val)'''
    a,b,te,ve = SGD(X_train, y_train, X_val, y_val, hidden_units, num_epoch, init_flag, learning_rate)
    #print("a",a)
    #print("b",b)
    #print("te",te)
    #print("ve",ve)
    #print("y_hat_train", y_hat_train)
    #print("y_hat_v",y_hat_v)
    train_predict = predict(a, b, X_train)
    #print("train_predict",train_predict)
    val_predict = predict(a, b, X_val)
    #print("val_predict",val_predict)
    train_e, val_e = error(train_predict, y_train, val_predict, y_val)
    #print(train_e, val_e)
    #print(ve.tolist())
    #print(te.tolist())

    textfile1 = open(train_out, "w")
    for elem in train_predict:
        textfile1.write(str(elem) + "\n")
    textfile1.close()

    textfile2 = open(validation_out, "w")
    for elem in val_predict:
        textfile2.write(str(elem) + "\n")
    textfile2.close()

    textfile3 = open(metrics_out, "w")
    for i in range(num_epoch):
        textfile3.write("epoch=%d crossentropy(train): %f\n"%(i+1, te[i]))
        textfile3.write("epoch=%d crossentropy(validation): %f\n"%(i+1, ve[i]))
    textfile3.write("error(train): %f\n" %train_e)
    textfile3.write("error(validation): %f\n" %val_e)
    textfile3.close()