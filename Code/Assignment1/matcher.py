import csv
import numpy as np

#Name: Jordy van der Tang

# PITCH
# This algorithm is perfect if you do not care about a portion of the participants getting a bad match!
# If you care about getting the top best matches to be linked together,
# this is for you. However if you are looking for a more balanced algorithm, look further!
# By greedly picking the best matches we make sure that if two people are a perfect match, we will match them together,
# even if it has a severe impact on everyone elses matches, we want that perfect match!

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


with open('icecream_responses.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile,delimiter=",", quotechar ="\"")
    header = reader.next()
    identifiers = []
    scores = {}
    for row in reader:
        identifier = row[0]
        ratings = map(int, [x if x is not '' else '0' for x in row[1:]])
        scores[identifier] = ratings

    # for key, value in scores.iteritems():
    #     print key,value
    #print "hi", type(scores)
    scorelist = scores.values()
    result = distance_matrix(scorelist)
    namelist = scores.keys()
    #print result
    matches = []
    matched = []
    for bl in range(len(scores)):
        curmax = 0
        maxi = 0
        for i in range(len(result)):
            if result[i][1] > curmax:
                if matched.count(result[i][0][0]) is 0 and matched.count(result[i][0][1]) is 0:
                    curmax = result[i][1]
                    maxi = i
        if curmax is 0 and maxi is 0:
            break
        else:
            matches.append(((namelist[result[maxi][0][0]],namelist[result[maxi][0][1]]), result[maxi][1]))
            matched.append(result[maxi][0][0])
            matched.append(result[maxi][0][1])
    print matches






