import numpy as np
import random as rnd
import string
import sys

# paths
entpath = sys.argv[1]
vecpath = sys.argv[2]
outpath = sys.argv[3]

# params
k = 10
n = 100
t = 1000

# load vectors
vec = np.load(open(vecpath,'r'))

# load entities
with open(entpath, 'r') as f:
    xp = f.read().splitlines()

# normalize
row_sums = np.square(vec).sum(axis=1)
norms = np.sqrt(row_sums)
vec = vec / norms[:,np.newaxis]

test_vec = vec[:t]
test_ent = xp[:t]

# choose random subset
#assert len(xp) == len(vec), "vectors not the same length as entity list"
#rnd.seed(0)
#test_idx = rnd.sample(range(len(xp)), n)
#test_vec = [vec[idx] for idx in test_idx]
#test_ent = [xp[idx] for idx in test_idx]

# find k-nearest neighbours to test set
f = open(outpath,'w')
for i,e in enumerate(test_ent):
    if i % 100 == 0:
	print("Test entity {}".format(i))
    v = test_vec[i]
    #d = np.linalg.norm(vec-v,axis=1)
    d = np.empty(vec.shape[0])
    for ind in range(vec.shape[0]):
        d[ind] = np.dot(v,vec[ind])
    min_idx = d.argsort()[-k:][::-1]
    ke = [xp[ii] for ii in min_idx]
    kd = [d[ii] for ii in min_idx]
    f.write('%s\n' % e)
    for idx, item in enumerate(ke):
	f.write('%s\t%f\n' % (ke[idx],kd[idx]))
    f.write('\n\n')
f.close()
