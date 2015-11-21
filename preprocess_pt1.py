import json
import re
import sys
import os
path = os.getcwd() + '/tweets/'

regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)+' # anything else
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

    hashtag_regex = re.compile("(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")
    tokens = ['' if hashtag_regex.match(token) else token for token in tokens]

    flag = False
    for item in tokens:
        if item=='rt':
            flag = True
            continue
        if flag and item=='@user':
            return ''
        else:
            flag = False

    return ' '.join([t for t in tokens if t]).replace('rt @user : ','')

metadata_list = [u'id', u'coordinates']

it = 0
with open('data/3/tweet_processed_text_en.txt', 'w') as tweet_processed_text, open('data/3/tweet_metadata_en.json', 'w') as metadata, open('data/3/tweet_hashtags.txt', 'w') as hashtag_f:
	for fname in os.listdir(path):
                f = open(path+fname,'r')
		for line in f:
			it = it + 1
			if it % 1000 == 0:
				print 'iteration %d' % it
	
	                try:
	                    data = json.loads(line)
	                except:
	                    continue
			if u'text' in data:
				if (not data.get(u'lang')) or data[u'lang'] != 'en':
					continue
				text = data[u'text']
				hashtags_text = []
				for hashtag in data[u'entities'][u'hashtags']:
					hashtags_text.append(hashtag[u'text'])
	
				
	
				# extract list of hashtags
				hashtags = set([re.sub(r"#+", "#", k) for k in set([re.sub(r"(\W+)$", "", j, flags = re.UNICODE) for j in set([i for i in text.split() if i.startswith("#")])])])
	
	
				line_text = preprocess(text) + '\n'
	                        if line_text=='\n':
	                            continue
	                        
	                        
				hashtag_f.write(json.dumps(hashtags_text) + '\n')
				tweet_processed_text.write(line_text.encode('utf8'))
	
				# write out metadata
				metadata_dict = {k:v for (k,v) in data.iteritems() if k in metadata_list}
				metadata_dict[u'text'] = line_text.encode('utf8')
				metadata_dict[u'user_id'] = data[u'user'][u'id']
				metadata_dict[u'hashtags'] = "|".join(hashtags).encode('utf8')
	
				json.dump(metadata_dict, metadata)
				metadata.write('\n')
