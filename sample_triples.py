import json
import random
import io
import sys

from collections import defaultdict

dictFile = sys.argv[1]
tripFile = sys.argv[2]
tripTagFile = sys.argv[3]

with io.open(dictFile,'r',encoding='utf-8') as fd, io.open(tripFile,'w',encoding='utf-8') as triplets, io.open(tripTagFile,'w',encoding='utf-8') as trip_hash:
	hashtag_dict = defaultdict(list)
	for line in fd:
		hashtag = json.loads(line)
		for tweet in hashtag[1]:
			hashtag_dict[hashtag[0]].append(tweet[1])


	hashtag_dict2 = dict(hashtag_dict)

	i = 0
	for hashtag in hashtag_dict.keys():
		i += 1
		if i % 10000 == 0:
			print("Iteration {}".format(i))
		curr_index = list(hashtag_dict[hashtag])
		for index in hashtag_dict[hashtag]:

			curr_index.remove(index)
			if curr_index:
				pos_index = random.choice(curr_index)
			else:
				continue

			triplets_array = []
			triplets_array.append(index)
			triplets_array.append(pos_index)

			triplets_hash = []
			triplets_hash.append(hashtag)
			triplets_hash.append(hashtag)

			loop_bool2 = False
			while not loop_bool2:
				neg_hashtag = random.choice(hashtag_dict2.keys())
				if neg_hashtag != hashtag:
					loop_bool2 = True
					neg_index = random.choice(hashtag_dict2[neg_hashtag])
					triplets_hash.append(neg_hashtag)
			
			triplets_array.append(neg_index)
			trip_hash.write(unicode(json.dumps(triplets_hash, ensure_ascii=False)) + '\n')
			triplets.write(unicode(json.dumps(triplets_array, ensure_ascii=False)) + '\n')
