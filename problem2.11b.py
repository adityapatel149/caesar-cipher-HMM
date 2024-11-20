from math import log
import numpy as np
from itertools import product


N = 2
minIters = 300
iters = 0
oldLogProb = float('-inf')
threshold = 0.0001



#Generate observation sequence from Ciphertext

ciphertext = open("ciphertext.txt", "r")
char_obs_seq = ciphertext.read()
ciphertext.close()

#Map characters to integers. stored as a dictionary
unique_chars = sorted(set(char_obs_seq))
char_to_int = {char:i for i,char in enumerate(unique_chars)}
#convert observation sequence to sequence of integers
obs_seq = [char_to_int[char] for char in char_obs_seq]


#number of characters
M = len(char_to_int)
# observatoin sequence length
T = len(obs_seq)

# Initialise Model
rand_values = np.random.uniform(-0.01,0.01, (N, N))
A = np.full((N,N), 1.0/N)
A += rand_values #add or subtract random values from all elements of A
A = np.clip(A, 0, None) # clip array A with minimum 0, maximum None (A should only have positive values)
A = A / A.sum(axis=1, keepdims=True) # normalise values in A so row sums to 1

rand_values = np.random.uniform(-0.01,0.01, (N, M))
B = np.full((N,M), 1.0/M)
B += rand_values
B = np.clip(B, 0, None)
B = B / B.sum(axis =1, keepdims = True)

rand_values = np.random.uniform(-0.1,0.1, N)
pi = np.full(N, 1.0/N)
pi += rand_values
pi = np.clip(pi, 0, None)
pi = pi / pi.sum()


#Scaling factors
c = np.zeros((T))


for iter in range(minIters):
    iters +=1 
    #Forward algorithm
    alpha = np.zeros((T,N))
    c[0] = 0
    for i in range(N):
        alpha[0, i] = pi[i] * B[i, obs_seq[0]]
        c[0] += alpha[0,i]

    c[0] = 1/c[0]

    for i in range(N):
        alpha[0, i] = c[0] * alpha[0, i]

    for t in range(1, T):
        c[t] = 0
        for i in range(N):
            sum_of_prev_alphas = 0
            for j in range(N):
                sum_of_prev_alphas += alpha[t-1, j] * A[j,i]

            alpha[t, i] = sum_of_prev_alphas * B [i, obs_seq[t]]
            c[t] += alpha[t, i]

        c[t] = 1/c[t]
        for i in range(N):
            alpha[t,i] = c[t] * alpha[t,i]



    #Backward algorithm:
    beta = np.zeros((T,N))
    for i in range(N):
        beta[T-1, i] = c[T-1]

    for t in range(T-2,-1,-1):
        for i in range(N):
            for j in range(N):
                beta[t,i] += A[i,j] * B[j, obs_seq[t+1]] * beta[t+1, j]
            beta[t,i] = c[t] * beta[t,i]



    #Calculate gamma and di-gamma
    gamma = np.zeros((T,N))
    digamma = np.zeros((T,N,N))
    for t in range(T-1):
        denom = 0
        for i in range(N):
            for j in range(N):
                denom += alpha[t,i] * A[i,j] * B[j, obs_seq[t+1]] * beta[t+1,j]
        
        for i in range(N):
            for j in range(N):
                digamma [t,i,j] = (alpha[t,i] * A[i,j] * B[j, obs_seq[t+1]] * beta[t+1, j]) / denom
                gamma[t,i] += digamma[t,i,j]

    denom = 0
    for i in range(N):
        denom += alpha[T-1, i]

    for i in range(N):
        gamma[T-1, i] = alpha[T-1,i]/denom


    # Re-estimate the model

        # Re-estimate pi
    for i in range(N):
        pi[i] = gamma[0, i]

        # Re-estimate A
    for i in range(N):
        for j in range(N):
            numer = 0.0
            denom = 0.0
            for t in range(T-1):
                numer += digamma[t,i,j]
                denom += gamma[t,i]
            A[i,j] = numer / denom

        # Re-estimate B
    for i in range(N):
        for j in range(M):
            numer = 0.0
            denom = 0.0
            for t in range(T):
                if(obs_seq[t] == j):
                    numer += gamma[t,i]
                denom += gamma[t,i]
        
            B[i,j] = numer/denom

    #Compute log(P(O|lamda))
    logProb = 0
    for i in range(T):
        logProb += log(c[i],10)
    logProb = -logProb

    print(iters)
    #To iterate or not
    delta = abs(logProb - oldLogProb)
    if delta > threshold:
        oldLogProb = logProb
        continue
    else:
        break




# print("iters \n", iters)
# print(N,M,T)
# print("prob \n", logProb)
# print("pi\n", pi)
# print("A \n", A)
# print("char_to_int", char_to_int)
print("B \n", B)