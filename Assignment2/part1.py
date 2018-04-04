import os
import numpy as np
import math
import matplotlib.pyplot as plt
import time

#the target distribution
sig =1
mu = 0
def p(x):
    return (math.exp(-1 * math.pow(x,4))*(2 + math.sin(5*x) + math.sin(-2*math.pow(x,2))))


def MetropolisHastings(sigma):
    #initial parameters
    x = -1
    N = 1500 #no of iterations

    samples = []
    samples.append(x)

    #main loop for the MH algorithm
    for i in range(N):
        #propose the conditional sample
        x_cand = np.random.normal(x, sigma)

        #compute the acceptance probability
        alpha = min(1 , p(x_cand)/p(x))

        #accept the sample with probability alpha
        u = np.random.uniform(0,1)
        if u < alpha :
            #accept the proposal
            x = x_cand
        # else reject the proposal and keep the older sample

        samples.append(x)

    samples = np.array(samples)
    return samples


def plot(s1,s2,s3):
    print("Plotting hist")
    fig, ax = plt.subplots(1, 3, sharey=False, tight_layout=True)
    ax[0].hist(s1,bins = 100)
    ax[1].hist(s2,bins = 100)
    ax[2].hist(s3,bins = 100)
    plt.title('Metropolis Hastings')
    plt.xlabel('Iterations')
    plt.ylabel('samples (x)')
    plt.show()

def main():
    # We are given 3 values of sigma. We will run the algorithm for all three values
    # and plot the histograms
    samples1 = MetropolisHastings(0.05)
    samples2 = MetropolisHastings(1)
    samples3 = MetropolisHastings(50)

    plot(samples1,samples2,samples3)


main()
