
from tkinter import X
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

hw = 8
N = hw ** 2
Sigma = 0.8
RateLearn = 0.3
field = hw
dimension = 2

fig = plt.figure()
ax = plt.axes()
plt.grid()
ax.set_aspect('equal')

x = np.zeros([N,dimension])
m = np.random.uniform(0,field,[N,dimension])
md = np.random.uniform(0,field,[N,dimension])

def squere(x):
    for i in range(N):
        for j in range(dimension):
            if j == 0:
                x[i][j] = i//hw
            else:
                x[i][j] = i%hw
    return x

def line(x):
    for i in range(N):
        for j in range(dimension):
            if j == 0:
                x[i][j] = 0
            else:
                x[i][j] = i
    return x

x = squere(x)
counter = 0

ims = []

while True:
    xtmps = []
    ytmps = []
    
    input = np.random.uniform(0,field,2)

    index_winner = 0
    value_min = field
    for i in range(N):
        tmp_box = 0
        for j in range(dimension):
            tmp_box += (input[j]-m[i][j])**2
        diff = np.abs(tmp_box**0.5)

        if diff < value_min:
            value_min = diff
            index_winner = i

    for i in range(N):
        for j in range(dimension):
            md[i][j] = RateLearn*(input[j] - m[i][j])*np.exp(-1*(np.abs(x[index_winner][j]-x[i][j])**2)/2*(Sigma**2))
        
    m += md

    # for i in range(N):
    #     for j in range(dimension):
    #         if j == 0:
    #             ytmps.append(m[i][j])
    #         else:
    #             xtmps.append(m[i][j])

    
    lines = []
    for i in range(N):

        for j in range(dimension):
            if j == 0:
                ytmps.append(m[i][j])
            else:
                xtmps.append(m[i][j])
        if i%hw == hw-1:
            im = plt.plot(ytmps,xtmps,marker="o",color="red")
            lines.extend(im)
    ims.append(lines)
    
    counter += 1
    if counter == 100:
        break

#plt.scatter(ytmps,xtmps)
ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=100)
plt.show()

# space = np.linspace(0, 2, 3)
# ax.set_xticks(space)
# ax.set_yticks(space)



# ani = animation.ArtistAnimation(fig, ims, interval=1, repeat_delay=1)
# ani.save('test.gif',writer='imagemagick')