# Forward + Viterbi Algorithms

The forward algorithm, in the context of a hidden Markov model (HMM), is used to calculate a 'belief state': the probability of a state at a certain time, given the history of evidence. The process is also known as filtering. The forward algorithm is closely related to, but distinct from, the Viterbi algorithm.

![Forward%20+%20Viterbi%20Algorithms%2036d74073fcb44060895fd9390e4661c9/Untitled.png](Forward%20+%20Viterbi%20Algorithms%2036d74073fcb44060895fd9390e4661c9/Untitled.png)

The Viterbi algorithm is a dynamic programming algorithm for obtaining the maximum a posteriori probability estimate of the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in the context of Markov information sources and hidden Markov models (HMM).

**Viterbi** - viterbi algorithm finds the best alignment for two sequences. 
**Forward** - The forward algorithm

Finds the best alignment score of s1  and s2
given two strands, initialEmission  matrix and initialTransition 

**plotCGGraph** - Includes the plot part for the viterbi algorithm

**ImportMatrix** - imports both of the emission and transition matrices