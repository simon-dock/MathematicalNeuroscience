import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

hw = 10
N = hw ** 2
RateLearn = 0.3
Iterator = 500
field = hw
dimension = 2

fig = plt.figure()
ax = plt.axes()
plt.grid()
ax.set_aspect('equal')

def square(x):
    rate = 10 / hw
    for i in range(N):
        for j in range(dimension):
            if j == 0:
                x[i][j] = rate * (i//hw +1)
            else:
                x[i][j] = rate * (i%hw+1)
    return x


def line(x):
    rate = 10 / hw
    for i in range(N):
        for j in range(dimension):
            if j == 0:
                x[i][j] = 0
            else:
                x[i][j] = rate * (i+1)
    return x


def triangle():

    while 1:
        x = np.random.uniform(-1,1, dimension)
        if x[0]<0:
            ymax=2*x[0]+1
        elif x[0]==0:
            ymax = 1
        else:
            ymax = -2*x[0]+1

        if x[1]<= ymax:
            break        
    return x

def cir():
    x = []
    r = np.random.uniform(0,1)
    rad = np.random.uniform(0, 2*np.pi)
    x.append(r*math.cos(rad))
    x.append(r*math.sin(rad))

    return x


def judge_winner(input,m):
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

    return index_winner


def update_ref(index_winner, input, x, m, counter):
    if counter/Iterator > 0.9:
        sigma = 0.1
    elif counter/Iterator > 0.8:
        sigma = 0.2
    elif counter/Iterator > 0.6:
        sigma = 0.4
    elif counter/Iterator > 0.5:
        sigma = 0.8
    elif counter/Iterator > 0.4:
        sigma = 1
    elif counter/Iterator > 0.3:
        sigma = 1.5
    elif counter/Iterator > 0.2:
        sigma = 2
    elif counter/Iterator > 0.1:
        sigma = 3
    else:
        sigma = 10
    for i in range(N):
        dist = 0
        for j in range(dimension):
            dist += (x[index_winner][j]-x[i][j])**2
        m[i] += RateLearn*(input - m[i])*np.exp(-dist/(2*sigma**2))
    
    return m


def draw_graph(m, ims, counter):
    lines = []
    xtmps = []
    ytmps = []
    for i in range(hw):
        for j in range(hw):
            for k in range(dimension):
                if k == 0:
                    ytmps.append(m[i+hw*j][k])
                else:
                    xtmps.append(m[i+hw*j][k])
        im = plt.plot(ytmps,xtmps,marker="o",color="red")
        lines.extend(im)
        ytmps=[]
        xtmps=[]

    for i in range(N):
        for j in range(dimension):
            if j == 0:
                ytmps.append(m[i][j])
            else:
                xtmps.append(m[i][j])
        if i%hw == hw-1:
            im = plt.plot(ytmps,xtmps,marker="o",color="red")
            lines.extend(im)
            ytmps=[]
            xtmps=[]
    
    lines.append(plt.text(0.0, 1.1, ("iteration = " + str(counter) + "/" + str(Iterator)), ha="center", va="bottom", fontsize="large"))
    ims.append(lines)

    return ims

def draw_line(m, ims, counter):
    lines = []
    xtmps = []
    ytmps = []

    for i in range(N):
        for j in range(dimension):
            if j == 0:
                ytmps.append(m[i][j])
            else:
                xtmps.append(m[i][j])
    lines.extend(plt.plot(ytmps,xtmps,color="red"))
    
    lines.append(plt.text(0.0, 1.1, ("iteration = " + str(counter) + "/" + str(Iterator)), ha="center", va="bottom", fontsize="large"))
    ims.append(lines)

    return ims


def main():
    x = np.zeros([N,dimension])
    m = np.random.uniform(-1,1,[N,dimension])
    #x = line(x)
    x = square(x)

    counter = 1
    ims = []

    while True:
        print(counter)
        #input = np.random.uniform(-1,1,2)
        input = triangle()
        #input = cir()   
        index_winner = judge_winner(input, m)
        m = update_ref(index_winner, input, x, m, counter)
        #ims = draw_line(m, ims, counter)
        ims = draw_graph(m, ims, counter)


        counter += 1
        if counter == Iterator:
            break

    ani = animation.ArtistAnimation(fig, ims, interval=1, repeat_delay=100)
    plt.show()
    ani = animation.ArtistAnimation(fig, ims, interval=1, repeat_delay=100)
    ani.save('try20.gif',writer='imagemagick')

if __name__ == "__main__":
    main()