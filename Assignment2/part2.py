import numpy as np
import math


# the NEIGHBOUR table
grid = [[],
        [2],[1, 3],[2, 4, 14],[3,15],[6, 16],[5, 7],[6, 8],[7, 9, 17],[8],[11],[10,12,19],[11],[],
        [3,15,21],[4,14,22],[5,23],[8,24],[26],[11,28],
        [21,31],[14,20,22],[15,21,32],[16,34],[17,25,36],[24,26,37],[18,25,27,38],[26,28],[19,27,39],[42],
        [31],[20,30],[22,33],[32,34],[23,33],[36],[24,35,37],[25,36,38],[26,37],[28,40],[39,41],[40,42],[29,31]]

#The transition matrix
# T(i,j) = prob of transition to state j at time t+1,
# given were at state i at time t
T = np.zeros(shape=(42,42))


# bit-wise error rate of the Sensor observations.
epsilon = 0.20


#The initial propabilities
# P(X0 = i) = 1/n
Pi =  np.full((42,1), 1./42)

#function to return the no of neighbours of a state
def N(s):
    return len(grid[s])

#function to compute the transition matrix
def calculateT():
    for i in range(42):
        for j in range(42):
            if j+1 in grid[i+1]:
                T[i][j] = 1./N(i+1)

#function to calculate the actual obstacles at state s
def calculateObstacles(s):
    neighbors = grid[s]
    obs = np.array([1,1,1,1])   # the obstacles as [N,W,E,S]
    if(len(neighbors) != 0):
        if(s-1 in neighbors):
            obs[1] = 0
        if(s+1 in neighbors):
            obs[2] = 0
        if(neighbors[0] <s-1):
            obs[0] = 0
        if(neighbors[-1] >s+1):
            obs[3] = 0

    return obs

#function to calculate the discrepancy btw actual reading and evidence
def calculateDiscrepancy(actual,evidence):
    return np.sum(actual!=evidence)


#function to compute the observation matrix
# it takes the evidence/observation variable as input and computes O
def calculateO(evidence):
    O = np.zeros(shape=(42,42))
    for i in range(42):
        d = calculateDiscrepancy(calculateObstacles(i+1),evidence)
        O[i][i] = math.pow(1 - epsilon,4-d) * math.pow(epsilon,d)
    return O


def FORWARD(f_t,e_tplus1):
    O = calculateO(e_tplus1)
    f_tplus1 = np.matmul(O,np.matmul(T.transpose(),f_t))
    return f_tplus1/np.sum(f_tplus1)

def logicalFiltering(evidences):
    f_t = Pi
    for i in range(len(evidences)):
        f_tplus1 = FORWARD(f_t,evidences[i])
        f_t = f_tplus1
        # print np.sum(f_tplus1)

    return f_t

calculateT()
f = logicalFiltering([[1,1,0,1],[1,0,0,1]])
print f

# print T
