import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from part2 import *

#function to calculate the coords of the states in the env
def calculateCoords():
    coords = {}
    for i in range(4):
        for j in range(16):
            if(env[i][j]!=-1):
                coords[env[i][j]] = [i,j]

    return coords

# function to generate a random Path in the grid of length 40.
# this random path can then be used to evaluate localization error and
# Viterbi path accuracy
def generatePath(len):
    # randomly get a start state
    path = []
    start = random.randint(1,42) #give a random number between 1 and 42

    #this loop to make sure we start with a state with some neighbors
    while(not grid[start]):
        start = random.randint(1,42)
    path.append(start)

    #generate the path by randomly jumping to neighbors
    for i in range(len-1):
        neighbors = grid[path[-1]]
        next = random.choice(neighbors)
        path.append(next)

    return path

#function to compute the actual obstacles for a given path
def generateEvidence(path):
    evidences =[]

    for i in range(len(path)):
        evidences.append(calculateObstacles(path[i]))

    return evidences

# function to randomly mutate the evidences according to value of epsilon
def errorEvidence(evidences, epsilon):
    for i in range(len(evidences)):
        for t in range(4):
            prob = random.random()
            if prob <= epsilon:
                evidences[i][t] = 1 - evidences[i][t]



def calculateManhattan(actual,estimated,coords):
    c1 = coords[actual]
    c2 = coords[estimated]

    diff = 0
    diff += abs(c1[0] - c2[0])
    diff += abs(c1[1] - c2[1])

    return diff

def ViterbipathAccuracy(actual, estimated):
    count =0.0
    for i in range(len(actual)):
        if actual[i] == estimated[i]:
            count +=1

    return count/len(actual)

def calculateStats(epsilon):
    coords = calculateCoords()
    error=[]
    accuracy = []
    for i in range(40):
        LE = 0.0
        PA = 0.0
        # if(i%10 == 0):
        #     print"done"
        for t in range(100):
            path =  generatePath(i+1)
            actualE = generateEvidence(path)

            mutatedE = copy.deepcopy(actualE)
            errorEvidence(mutatedE,epsilon)

            actual = path[-1]
            estimated = np.argmax(logicalFiltering(mutatedE))+1

            estimatedPath = Viterbi(mutatedE)

            LE += calculateManhattan(actual,estimated,coords)
            PA += ViterbipathAccuracy(path,estimatedPath)
        LE /= 100.0
        PA /= 100.0
        error.append(LE)
        accuracy.append(PA)
    return (error,accuracy)

def plotter():
    epsilon = [0.00,0.02,0.05,0.10,0.20]
    f= plt.figure(1)
    plt.margins(0.05, 0.1)

    g= plt.figure(2)
    plt.margins(0.05, 0.1)

    for e in epsilon:

        LE,  PA =calculateStats(e)
        plt.figure(1)
        plt.plot(range(40),LE)

        plt.figure(2)
        plt.plot(range(40),PA)
        print "done OKAY"

    plt.figure(1)
    plt.legend(['e = 0.00', 'e = 0.02', 'e = 0.05', 'e = 0.10','e = 0.20'], loc='upper right')

    plt.figure(2)
    plt.legend(['e = 0.00', 'e = 0.02', 'e = 0.05', 'e = 0.10','e = 0.20'], loc='lower right')

    f.savefig('LE.png')
    g.savefig('PA.png')
    plt.show()

if __name__ == "__main__":
    plotter()
