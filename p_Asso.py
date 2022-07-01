import numpy as np
import matplotlib.pyplot as plt
import copy

from yaml import safe_dump
n = 1000
m = 80
start = 100
q = 10
rate = 100

np.random.seed(5)

w = np.zeros([n,n])         
x = np.random.choice([-1,1], size=(m,n), p=[0.5,0.5])
V_x = copy.deepcopy(x.T)
H_x = copy.deepcopy(x)
w = np.dot(V_x, H_x)
w = w/n
v = np.zeros(n)
np.fill_diagonal(w,v)

for count in range(q-1):
    t, base_t = copy.deepcopy(x[0]), copy.deepcopy(x[0])
    alpha = int(count*rate+start)
    for i in range(alpha):
        t[i] = base_t[alpha-(i+1)]
    
    round = 0
    datalist = []

    for i in range(20):
        datalist.append((np.dot(t,x[0]))/n)
        t = np.copy(np.sign(np.dot(w,t)))
    plt.plot(datalist,marker="o")

plt.xlim(0,20)
plt.show()