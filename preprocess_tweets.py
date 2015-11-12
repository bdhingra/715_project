import json
import re
import random

regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    tokens = [token.lower() for token in tokens]

    html_regex = re.compile('<[^>]+>')
    tokens = [token for token in tokens if not html_regex.match(token)]

    mention_regex = re.compile('(?:@[\w_]+)')
    tokens = ['@user' if mention_regex.match(token) else token for token in tokens]

    url_regex = re.compile('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
    tokens = ['!url' if url_regex.match(token) else token for token in tokens]

    num_regex = re.compile('(?:(?:\d+,?)+(?:\.?\d+)?)')
    tokens = ['###' if num_regex.match(token) else token for token in tokens]

    return ' '.join(tokens).replace('rt @user : ','')

metadata_list = [u'id', u'coordinates']

it = 0
with open('tweet_text_en.txt', 'w') as tweet_text:
    with open('tweet_metadata_en.json', 'w') as metadata:
    	with open('tweet_sample.txt') as f:
			lines = f.read().splitlines()
			hashtagMap = {}
			lines1 = lines
			print(len(lines))
			for line in lines1:
				data = json.loads(line)
				if data.get(u'delete'):
					lines.remove(line)
				else:
					if not data[u'user'][u'lang'] == 'en':
						lines.remove(line)
					else: 
						if(data[u'entities'][u'hashtags'] == []):
					 		lines.remove(line)
						else:
							for hashtag in data[u'entities'][u'hashtags']:
								if hashtag[u'text'] in hashtagMap.keys():
									hashtagMap[hashtag[u'text']].append(data[u'text'])
								else:
									hashtagMap[hashtag[u'text']] = [data[u'text']]
			lines2 = lines
			print(len(lines))
			for line in lines2:
				data = json.loads(line)
				if(not data.get(u'entities')):
					lines.remove(line)
				else:
					for hashtag in data[u'entities'][u'hashtags']:
						if not hashtagMap.get(hashtag[u'text']):
							if(line in lines):
								lines.remove(line)
						elif len(hashtagMap[hashtag[u'text']]) == 1:
							if(line in lines):
								lines.remove(line)

			it = it + 1
			if it % 1000 == 0:
			    print 'iteration %d' % it

			print(hashtagMap)
			print(len(lines))
			print(len(hashtagMap))
			for line in lines:
				data = json.loads(line)
				text = data[u'text']

				hashtags_metadata = set([re.sub(r"#+", "#", k) for k in set([re.sub(r"(\W+)$", "", j, flags = re.UNICODE) for j in set([i for i in text.split() if i.startswith("#")])])])


				line_text = preprocess(text) + '\n'
				if data[u'entities'][u'hashtags']:
					print('here')
					test_hashtag = random.choice(data[u'entities'][u'hashtags'])
					if hashtagMap.get(test_hashtag[u'text']):
						tweet_text.write(line_text.encode('utf8'))
						text_pos = random.choice(hashtagMap[test_hashtag[u'text']])
						loop_bool = False

						while not loop_bool:
							neg_hashtag = random.choice(hashtagMap.keys())
							if not(neg_hashtag == test_hashtag[u'text']):
								loop_bool = True
								text_neg = random.choice(hashtagMap[neg_hashtag])
							
						pos_line_text = preprocess(text_pos) + '\n'
						tweet_text.write(pos_line_text.encode('utf8'))
						neg_line_text = preprocess(text_neg) + '\n'
						tweet_text.write(neg_line_text.encode('utf8'))

						# write out metadata
						metadata_dict = {k:v for (k,v) in data.iteritems() if k in metadata_list}
						metadata_dict[u'text'] = line_text.encode('utf8')
						metadata_dict[u'user_id'] = data[u'user'][u'id']
						metadata_dict[u'hashtags'] = "|".join(hashtags_metadata).encode('utf8')

						json.dump(metadata_dict, metadata)
						metadata.write('\n')

	



