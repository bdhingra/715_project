import random
import json

lines = open('data/sample_10M/sample_triples_100K.txt').readlines()
random.shuffle(lines)
with open('data/sample_10M/randomized_triplets_100K.txt', 'w') as f:
	for line in lines:
		triplets = json.loads(line)
		for el in triplets:
			f.write(el.encode('utf8') + '\n')
