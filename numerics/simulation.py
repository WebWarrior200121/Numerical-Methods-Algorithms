import numpy as np
from numpy import random

# Function to generate uniform random numbers using LCG
def randomint(length, seed=0, param={"a":4, "c":1, "M":9}):

    r = seed
    a, c, M = param.values()
    random_list = [r]
    
    for i in range(length-1):
        r = (a*r+c)%M
        random_list.append(r)      
    
    return random_list
    
    
# Function to generate two Gaussian random numbers
def random_gaussian(mu=0, sigma=1):
    
    r = np.sqrt(-2*np.log(1 - random.random()))
    
    theta = 2*np.pi*random.random()
    
    x = mu + sigma*r*np.cos(theta)
    y = mu + sigma*r*np.sin(theta)
    
    return x, y


# Function to generate steps of a 2d random walk
def random_walk(n):
    steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    positions = steps[random.choice([0, 1, 2, 3], n)]
    positions = np.concatenate(([[0, 0]], positions))
    path = positions.cumsum(axis=0)
    return path
    
    
# Function to simultate simple nuclei decay
def nuclei_decay_simul(n, tau, dt, tmax):
    # time
    time = np.arange(0., tmax, dt)
    
    # initial populations
    pop = []

    # decay probability
    p = 1 - 2**(-dt/tau)

    # decay simulation
    for t in time:
        pop.append(n)

        # counting atoms that decay
        decay_count = 0
        
        for i in range(n):
            if (random.random() < p): 
                decay_count += 1
        
        n -= decay_count
        
    return time, np.array(pop)


# Function to integrate using hit or miss method 
def hit_or_miss_int(f, box, n=100000, save=False):
    count = 0
    area = np.prod(np.diff(box))
    
    ymin = box[1][0] 
    x = random.uniform(*box[0], n)
    y = random.uniform(*box[1], n)
    
    if save == True:
        npts = np.arange(1, n + 1)
        count = np.cumsum((ymin < y) & (y < f(x)))
        result = npts, area*count/npts
    else:
        count = np.sum((ymin < y) & (y < f(x)))
        result = area*count/n
        
    return result

def hit_or_miss(f, box, n=100000, save=False):
    count = 0
    area = np.prod(np.diff(box))
    
    ymin = box[1][0] 
    #x = random.uniform(*box[0], n)
    #y = random.uniform(*box[1], n)
    
    for i in range(n):
        x = random.uniform(*box[0])
        y = random.uniform(*box[1])
        
        if ymin <= y and y <= f(x):
            count += 1
    
    return area * count / n

def hit_or_miss_mod(f, box, n=100000, axis = 0):
    count = 0
    area = np.prod(np.diff(box))
    area = abs(area)
    
    #ymin = box[1][0] 
    #x = random.uniform(*box[0], n)
    #y = random.uniform(*box[1], n)
    
    for i in range(n):
        x = np.random.uniform(*box[0])
        y = np.random.uniform(*box[1])
        
        if y >= axis:
            if y <= f(x):
                #plt.plot(x,y,'o',color ='b')
                count += 1
        
        if y <= axis:
            if y >= f(x):
                #plt.plot(x,y,'o', color = 'r')
                count -=1
    
    return area * count / n


def mean_value_int(f, box, n = 10**(5)):
    
    hyper_volume = np.prod([np.diff(b) for b in box])
    
    random_points = np.array([np.random.uniform(b[0],b[1],n) for b in box])
    
    mean_value = 0
    for i in range(n):
        #print(random_points[:,i])
        f_value = f(random_points[:,i]) #f recieves an array. 3d example: [-0.10489765  0.81583276  1.52275235]
        mean_value += f_value
    
    
    mean_value = mean_value / n

    integral = hyper_volume * mean_value
    
    return integral

def mean_value_int2(f, box, n = 10**(5)):
    
    hyper_volume = np.prod([np.diff(b) for b in box])
    
    random_points = np.array([np.random.uniform(b[0],b[1],n) for b in box])
    
    f_values = f(random_points)

    
    
    mean_value = np.mean(f_values)

    integral = hyper_volume * mean_value
    
    return integral

def importance_sampling_int(g,F,W_int,n = 10**7):
    
    random_points = F(n)
    g_values = g(random_points)
    mean_value = np.mean(g_values)
    integral = mean_value * W_int
    return integral

def importance_sample_int2(g,F,W_int,n=10**7):
    expval = 0
    random_points = F(n)
    for i in range(n):
        expval += g(random_points[i])
    
    expval = expval / n
    integral = expval * W_int
    
    return integral

def importance_sample(g,F,W,n=10**7):
    # W = integral(w)
    # x = F(z) , z in [0,1]
    
    expval = 0
    for i in range(n):
        z = np.random.uniform(0,1)
        x = F(z)
        g_val = g(x)
        expval += g_val
    
    expval = expval / n
    integral = expval * W
    
    return integral


def rand_direction():
    phi = 2 * np.pi * np.random.random()
    theta = np.arccos(1 - 2 * np.random.random())
    return phi, theta


def random_walker_trial(steps):
    x = 0.0
    y = 0.0
    z=0.0
    
    for _ in range(steps):
        phi, theta = rand_direction()
        x += np.sin(theta) * np.cos(phi)
        y += np.sin(theta) * np.sin(phi)
        z += np.cos(theta)
        
    distance = np.sqrt(x**2 + y**2 + z**2)
    
    return distance,x

def random_walker(trials, steps):
    average_distance = 0
    average_x = 0
    for _ in range(trials):
        distance,x = random_walker_trial(steps)
        average_distance += distance
        average_x += x
    
    average_distance /= trials
    average_x /= trials
    
    return average_distance,average_x


def discrete_nuclei_decay(N,p,T,M):
    dt = T/M
    #print("dt: ", dt)
    P = p * dt
    #print("Prob: ",P)
    #print()
    
    N_data = np.array([N])
    T_data = np.array([dt * i for i in range(M)])
    
    N_decay = 0
    N = N - N_decay
    for i in range(1,M):
        N_decay = 0
        for _ in range(N):
            if np.random.random() < P:
                N_decay +=1
        N = N - N_decay
        add = np.array([N])
        N_data = np.concatenate((N_data, add))
        
        #print("time",T_data[i])
        #print("decay: ", N_decay)
        #print("remain: ", N)
        #print( ) 
    
    
    return T_data,N_data


def discrete_nuclei_decay_exp():
    N = 100
    p = 0.2
    T = 20
    M = 100
    prob = print("probability: ", p * T / M)
    T_data, N_data = discrete_nuclei_decay(N,p,T,M)
    plt.plot(T_data,N_data,'.')
    plt.plot(T_data, N * np.exp(-p * T_data))
    

def mean_value_3d(f,box,n = 10**7):
    vol = np.prod(np.diff(box))
    expval = 0
    
    for _ in range(n):
        x = np.random.uniform(*box[0])
        y = np.random.uniform(*box[1])
        z = np.random.uniform(*box[2])
        fval = f(x,y,z)
        expval += fval
    
    expval = expval/n
    return expval * vol

def mean_value_2d(f,box,n = 10**7):
    vol = np.prod(np.diff(box))
    expval = 0
    
    for _ in range(n):
        x = np.random.uniform(*box[0])
        y = np.random.uniform(*box[1])
        fval = f(x,y)
        expval += fval
    
    expval = expval/n
    return expval * vol


def mean_value_1d(f,box,n = 10**7):
    vol = np.prod(np.diff(box))
    expval = 0
    
    for _ in range(n):
        x = np.random.uniform(*box[0])
        #y = np.random.uniform(*box[1])
        fval = f(x)
        expval += fval
    
    expval = expval/n
    return expval * vol
    
