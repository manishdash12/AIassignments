# Sampling and Localization
This assigment has two parts.

Part 1 implements the Metropolis Hastings algorithm for sampling data from a given distribution.

Part 2 solves the localization example given in section 3.2 of "Artificial Intelligence: A modern approach", Third edition by Russell and Norvig (pages 591-593)

### Metropolis Hastings
The code can be run by :
  ```
  python part1.py
  ```
As given in the assignment, 1500 samples are taken. If we increase the number of samples to the order of 100,000 then the distribution almost resembles the target distribution.

### Robot localization
* **part2.py** has the code for the main functions: logicalFiltering and Viterbi algorithm.
* **part2b.py** has the code for evaluating the Localization error and Viterbi path accuracy for a large number of randomised experiments.
