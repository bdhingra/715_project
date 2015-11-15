import random
import json

lines = open('triplets.txt').readlines()
random.shuffle(lines)
with open('randomized_triplets.txt', 'w') as f:
	for line in lines:
		triplets = json.loads(line)
		for el in triplets:
			f.write(el.encode('utf8') + '\n')