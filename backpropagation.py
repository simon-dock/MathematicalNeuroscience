import numpy as np
import matplotlib.pyplot as plt
from sympy import primitive

def sigmoid(x):
    return 1/(1+np.exp(-x))

def der_sigmoid(x):
    return x*(1-x)

def squared_error(z, t):
    return (z-t)**2/2

#np.random.seed(20220512)

epoch = 10000
lr = 0.7
I, H, O = 3, 2, 1

#inputs= np.array([ [1,0,0,0,0,0,0], [1,0,0,0,0,0,1], [1,0,0,0,0,1,0], [1,0,0,0,0,1,1], [1,0,0,0,1,0,0], [1,0,0,0,1,0,1], [1,0,0,0,1,1,0], [1,0,0,0,1,1,1], [1,0,0,1,0,0,0], [1,0,0,1,0,0,1], [1,0,0,1,0,1,0], [1,0,0,1,0,1,1], [1,0,0,1,1,0,0], [1,0,0,1,1,0,1], [1,0,0,1,1,1,0], [1,0,0,1,1,1,1], [1,0,1,0,0,0,0], [1,0,1,0,0,0,1], [1,0,1,0,0,1,0], [1,0,1,0,0,1,1], [1,0,1,0,1,0,0], [1,0,1,0,1,0,1], [1,0,1,0,1,1,0], [1,0,1,0,1,1,1], [1,0,1,1,0,0,0], [1,0,1,1,0,0,1], [1,0,1,1,0,1,0], [1,0,1,1,0,1,1], [1,0,1,1,1,0,0], [1,0,1,1,1,0,1], [1,0,1,1,1,1,0], [1,0,1,1,1,1,1], [1,1,0,0,0,0,0], [1,1,0,0,0,0,1], [1,1,0,0,0,1,0], [1,1,0,0,0,1,1], [1,1,0,0,1,0,0], [1,1,0,0,1,0,1], [1,1,0,0,1,1,0], [1,1,0,0,1,1,1], [1,1,0,1,0,0,0], [1,1,0,1,0,0,1], [1,1,0,1,0,1,0], [1,1,0,1,0,1,1], [1,1,0,1,1,0,0], [1,1,0,1,1,0,1], [1,1,0,1,1,1,0], [1,1,0,1,1,1,1], [1,1,1,0,0,0,0], [1,1,1,0,0,0,1], [1,1,1,0,0,1,0], [1,1,1,0,0,1,1], [1,1,1,0,1,0,0], [1,1,1,0,1,0,1], [1,1,1,0,1,1,0], [1,1,1,0,1,1,1], [1,1,1,1,0,0,0], [1,1,1,1,0,0,1], [1,1,1,1,0,1,0], [1,1,1,1,0,1,1], [1,1,1,1,1,0,0], [1,1,1,1,1,0,1], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1] ])
#teachers = np.array([1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1]) 

inputs = np.array(([1,0,0],[1,0,1],[1,1,0],[1,1,1]))
teachers = np.array([0,1,1,0])

hiddens = np.zeros(H)
output = np.zeros(O)

S = np.random.normal(0.0, 0.1, (I,H))
W = np.random.normal(0.0, 0.1, (H+1,O))

pers_h = []
h_x = []
h_x_per = []
for k in range(1):
    count = 0
    errors = np.zeros(len(inputs))
    error_aves = []
    errors = []

    pers = []
    cc = 0

    while(1):

        i = np.random.randint(0,4)
        count += 1
        error = 0
        input = inputs[i]
        teacher = teachers[i]
        hiddens = np.dot(input, S)
        hiddens = sigmoid(hiddens)
        hiddens_tmp = np.concatenate([[1],hiddens])
        output = np.dot(hiddens_tmp,W)
        output = sigmoid(output)
        error = squared_error(output,teacher)
        r = (teacher - output)*output*(1-output)
        delta_w =  lr * r * hiddens_tmp
        r_dash =  r * W[1:].T* hiddens * (1-hiddens)
        delta_s =  lr * r_dash.T * input
        delta_w = delta_w.reshape(H+1,O)
        delta_s = delta_s.T
        W += delta_w
        S += delta_s 
        print(W[0])

        error_aves.append(error)
        h_x.append(count)

        if count % 10 == 11:
            cc += 1
            for i in range(len(inputs)):
                input = inputs[i]
                teacher = teachers[i]
                hiddens = np.dot(input, S)
                hiddens = sigmoid(hiddens)
                hiddens_tmp = np.concatenate([[1],hiddens])
                output = np.dot(hiddens_tmp,W)
                output = sigmoid(output)
                m = teacher - output
                if m < 0 :
                    per = 1 + m
                else:
                    per = 1 - m
                pers.append(per)
            pers_h.append(np.mean(pers))
            h_x_per.append(cc)


        errors.append(error)
        if count % 4 == 3:
            if np.mean(errors) < 0.001:
                break
            errors = []

        if count > 10000000:
            break

figure = plt.axes()
plt.title("error")
plt.xlabel('iterations')
figure.grid()
plt.scatter(h_x,error_aves, s=1,color="blue")
#plt.scatter(h_x_per,pers_h, s=1, color="red")
#plt.ylim(0.0,1.0)
plt.show()