import json
import random
import io

def getline(desired_line_number):
    if desired_line_number < 1: return ''
    for current_line_number, line in enumerate(open('data/tweet_processed_text_en.txt')):
        if current_line_number == desired_line_number-1: return line.decode('utf8')
    return ''


with open('data/tweet_hashtags.txt') as hashtag_f, io.open('data/triplets.txt', 'w', encoding='utf8') as triplets, io.open('data/triplets_hashtags.txt','w', encoding='utf-8') as trip_hash:

	hashtag_dict = {}
	i = 1
	for line in hashtag_f:
		hashtags = json.loads(line)
		for hashtag in hashtags:
			if hashtag in hashtag_dict.keys():
				hashtag_dict[hashtag].append(i)
			else:
				hashtag_dict[hashtag] = [i]
		i = i+1

	#for hashtag in hashtag_dict.keys():
	#	if len(hashtag_dict[hashtag]) == 1:
	#		hashtag_dict.pop(hashtag, None)


	hashtag_dict2 = dict(hashtag_dict)

	for hashtag in hashtag_dict.keys():
                curr_index = list(hashtag_dict[hashtag])
                for index in hashtag_dict[hashtag]:
			#triplets_array.append(text_ex.strip('\n'))

                        curr_index.remove(index)
                        if curr_index:
                            pos_index = random.choice(curr_index)
                        else:
                            continue
			#loop_bool = False

			#while not loop_bool:
			#	pos_index = random.choice(hashtag_dict2[hashtag])
			#	if pos_index != index:
			#		loop_bool = True


			text_ex = getline(index)
			triplets_array = []
			#triplets_array.append(text_ex.encode('utf-8').strip('\n'))
			triplets_array.append(text_ex.strip('\n'))
			#triplets_array.append(getline(pos_index).encode('utf-8').strip('\n'))
			triplets_array.append(getline(pos_index).strip('\n'))

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
			
			#triplets_array.append(getline(neg_index).encode('utf-8').strip('\n'))
			triplets_array.append(getline(neg_index).strip('\n'))
                        trip_hash.write(unicode(json.dumps(triplets_hash, ensure_ascii=False)) + '\n')
			triplets.write(unicode(json.dumps(triplets_array, ensure_ascii=False)) + '\n')

