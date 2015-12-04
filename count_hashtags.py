import sys
import json
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

dictPath = sys.argv[1]
N = int(sys.argv[2])

# create dictionary of hashtags and tweet counts
counts = defaultdict(int)
total = 0
with open(dictPath,'r') as d:
	for line in d:
		row = json.loads(line)
		counts[row[0]] = len(row[1])
		total += len(row[1])

# sort in descending order
sorted_tags = sorted(counts, key=lambda k:counts[k], reverse=True)
sorted_counts = [counts[ii] for ii in sorted_tags]

# print top N
print("Top - ")
for i in range(N):
	print("{} - {}".format(sorted_tags[i],sorted_counts[i]))

# print bottom N
print("Bottom - ")
for i in range(1,N+1):
	print("{} - {}".format(sorted_tags[-i],sorted_counts[-i]))

# tweets above count 5
s = 0
for c in sorted_counts:
	if c > 50:
		s+=c
print("Number of tweets above threshold = {} (total {})".format(s,total))

# plot values
h,b = np.histogram(sorted_counts, bins=500)
plt.figure()
plt.plot(h)
plt.yscale('log')
#plt.hist(sorted_counts,bins=10000)
#plt.yscale('log')
plt.show()
