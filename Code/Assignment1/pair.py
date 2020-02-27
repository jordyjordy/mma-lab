import math
import numpy as np

print "Exercise 1 - pairer"

def euclidian_distance(p,q):
    d = 0
    for x in range(len(p)):
        d += (p[x]-q[x])**2
    return math.sqrt(d)

def cosine_similarity(p, q):
    # print(p)
    # print(q)
    numer = np.sum(np.multiply(p,q))
    denom = np.sqrt(np.sum(np.square(p)))*np.sqrt(np.sum(np.square(q)))
    if(denom is not 0):
        return float(numer)/denom
    else:
        return 0.0

def normalize_vector(p,maximum):
    return [float(x)/ maximum for x in p]

def distance_matrix(vectors):
    result = []
    N = len(vectors)
    for i in range(N):
        for j in range(i+1,N):
            ind = (i,j)
            dis = cosine_similarity(vectors[i],vectors[j])
            result.append((ind,dis))
    return result


