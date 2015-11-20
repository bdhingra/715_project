import numpy as np
import json
import sys
from sklearn.cluster import KMeans

num_clusters = int(sys.argv[1]) 
encoded_file = sys.argv[2]
processed_text_file = sys.argv[3]
topics_file = sys.argv[4]

# Read encoded feature file
features = np.load(encoded_file)

# Read processed text file
with open(processed_text_file) as f:
    tweets = np.array(f.read().splitlines())
    
# Read groundtruth topics file
with open(topics_file) as f:
    topics = json.load(f)

# Run K-Means to find cluster labels
estimator = KMeans(n_clusters=num_clusters)
estimator.fit(features)
labels = estimator.labels_

# Print cluster labels for groundtruth examples
for topic, examples in topics.iteritems():
    print topic
    labeled_tweets = []
    for ex in examples:
        labeled_tweets.append((labels[ex], tweets[ex]))
    labeled_tweets.sort(key = lambda tup: tup[0])
    for (cluster, tweet) in labeled_tweets:
        print cluster, tweet
    print '\n'
