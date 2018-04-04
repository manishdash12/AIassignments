import os
import numpy as np
import math
import matplotlib.pyplot as plt


#the target distribution
def p(x):
    return (math.exp(-math.pow(x,4))*(2 + math.sin(5*x) + math.sin(-2*math.pow(x,2))))


def MetropolisHastings(sigma):
    #initial parameters
    x = -1
    N = 1500 #no of iterations

    samples = []
    samples.append(x)
    #draw random samples from the proposal distribution
    q = np.random.normal(x, sigma, 1500)

    #main loop for the MH algorithm
    for i in range(1500):
        #propose the conditional sample
        x_cand = x + q[i]

        #compute the acceptance probability
        alpha = min(1, (p(x_cand)/p(x)))

        #draw an uniform sample
        u = np.random.uniform(0,1)

        if u < alpha:
            #accept the proposal
            x = x_cand
        # else reject the proposal and keep the older sample

        samples.append(x)

    return samples


def plot(samples):
    print("Plotting hist")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,)
    ax.hist(samples, bins=1000)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

def main():
    #We are given 3 values of sigma. We will run the algorithm for all three values
    # and plot the histograms

    for sigma in [0.05, 1, 50]:
        plot(MetropolisHastings(sigma))

main()
