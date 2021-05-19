N  = 5 # power of 10, 10^N is the number of trials conducted

import numpy as np

np.random.seed = 10

def first_a(n):
    count = 0
    for i in range(n):
        results = np.random.binomial(1, .5, 7)
        count = count + (np.sum(results[:4])==3)
    return count/n

def first_c(n):
    countA = 0
    countAB = 0
    for i in range(n):
        results = np.random.binomial(1, .5, 7)
        countA = countA + (np.sum(results[:4])==3)
        countAB = countAB + (np.sum(results[:4])==3 and np.sum(results[4:7])==0)
    return countAB/countA

def first_e(n):
    countA = 0
    countAB = 0
    for i in range(n):
        results = np.random.binomial(1, [.75, .75, .25, .25], 4)
        countA = countA + (np.sum(results)==3)
        countAB = countAB + (np.sum(results)==3 and np.sum(np.random.binomial(1,[.75, .25, .75], 3))==0)
    return countAB/countA


if __name__ == '__main__':
    n = np.power(10,N)
    print("For N = "+ str(n)+" , the simulated value for part(a) is "+str(first_a(n)))
    print("For N = "+ str(n)+" , the simulated value for part(c) is "+str(first_c(n)))
    print("For N = "+ str(n)+" , the simulated value for part(e) is "+str(first_e(n)))