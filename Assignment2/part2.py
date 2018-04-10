import numpy as np
import math


# the NEIGHBOUR table
grid = [[],
        [2],[1, 3],[2, 4, 14],[3,15],[6, 16],[5, 7],[6, 8],[7, 9, 17],[8],[11],[10,12,19],[11],[],
        [3,15,21],[4,14,22],[5,23],[8,24],[26],[11,28],
        [21,31],[14,20,22],[15,21,32],[16,34],[17,25,36],[24,26,37],[18,25,27,38],[26,28],[19,27,39],[42],
        [31],[20,30],[22,33],[32,34],[23,33],[36],[24,35,37],[25,36,38],[26,37],[28,40],[39,41],[40,42],[29,31]]

num_states = 42
#The transition matrix
# T(i,j) = prob of transition to state j at time t+1,
# given were at state i at time t
T = np.zeros(shape=(num_states,num_states))


# bit-wise error rate of the Sensor observations.
epsilon = 0.20


#The initial propabilities
# P(X0 = i) = 1/n
Pi =  np.full((num_states,1), 1./num_states)

#function to return the no of neighbours of a state
def N(s):
    return len(grid[s])

#function to compute the transition matrix
# the assumption is that the probability of transitioning to any of the neighbors
# is uniform. Also the robot is required to move to one of the neighbors,
# so T[i][i]=0
def calculateT():
    for i in range(num_states):
        for j in range(num_states):
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
# the evidence matrix O is a diagnol matrix where
# O[i][i] = P(E_t | X_t = i)
def calculateO(evidence):
    O = np.zeros(shape=(num_states,num_states))
    for i in range(num_states):
        d = calculateDiscrepancy(calculateObstacles(i+1),evidence)

        # this is specified in the problem statement
        O[i][i] = math.pow(1 - epsilon,4-d) * math.pow(epsilon,d)
    return O


# this function computes the the vector f(1:t+1) from f(1:t) and O_tplus1
def FORWARD(f_t,e_tplus1):
    O = calculateO(e_tplus1)
    f_tplus1 = np.matmul(O,np.matmul(T.transpose(),f_t))

    #normalise the vector( multiply by alpha)
    return f_tplus1/np.sum(f_tplus1)

# this function implements the forward algorithm a.k.a. Logical Filtering
def logicalFiltering(evidences):
    f_t = Pi

    #calculate the forward vectors f(1:t) incrementally
    for i in range(len(evidences)):
        f_tplus1 = FORWARD(f_t,evidences[i])
        f_t = f_tplus1

    return f_t


# This function implements the Viterbi algorithm to approximate the most likely
# explanation of the evidence
# It is a dynamic programming algorithm and returns the most likely state-path
# algorithm as in : https://courses.engr.illinois.edu/cs447/fa2017/Slides/Lecture07.pdf
def Viterbi(evidence):
    num_obs = len(evidence)

    # the dynamic programming vectors
    # viterbi :store the probabilities of the most likely path upto state i
    # backtrack : store the preceeding state in the most likely path
    # together they form the Trellis matrix
    viterbi = np.zeros(shape=(num_states,num_obs))
    backtrack = np.zeros(shape=(num_states,num_obs),dtype=int)

    #initialise the trellis matrix for the first evidences
    O = calculateO(evidence[0])
    viterbi[:,0] = np.matmul(O,Pi)[:,0]

    #for the rest of the evidence populate the Trellis matrix
    for i in range(1,num_obs):
        for t in range(num_states):
            tmp = np.multiply(viterbi[:,i-1],T[:,t])
            #extract the best path
            viterbi[t][i] = np.amax(tmp)
            backtrack[t][i] = np.argmax(tmp)

        #update the Trellis matrix
        O = calculateO(evidence[i])
        viterbi[:,i] = np.matmul(O,viterbi[:,i])

    # Now to compute the best path
    #First, extract the last state where the path ends
    t_max = np.argmax(viterbi[:,num_obs - 1])
    vit_max = np.max(viterbi[:,num_obs - 1])

    #Using the backtrack vector, compute the path recursively
    i = num_obs -1
    path = np.zeros(shape=(1,num_obs))
    while(i>=0):
        path[0][i] = t_max
        t_max = backtrack[t_max][i]
        i = i-1

    return path

if  __name__ == "__main__":
    calculateT()

    f = logicalFiltering([[1,1,0,1]])

    # make the probabilities vector an array
    f =np.array(f[:,0])

    #extract the best 5 positions in the array
    # print f.argsort()[-5:][::-1]

    print Viterbi([[1,1,0,1]])
