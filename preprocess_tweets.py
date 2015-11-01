import json
import re

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
	with open('tweets.2014-09-30T22_42_47') as f:
	    for line in f:
		it = it + 1
		if it % 1000 == 0:
		    print 'iteration %d' % it

		data = json.loads(line)
		if u'text' in data:
		    if data[u'lang'] != 'en':
			continue
		    text = data[u'text']

		    # extract list of hashtags
		    hashtags = set([re.sub(r"#+", "#", k) for k in set([re.sub(r"(\W+)$", "", j, flags = re.UNICODE) for j in set([i for i in text.split() if i.startswith("#")])])])
		    

		    # insert whitespace around chains of non-alphanumeric characters
		    #tokenized = re.sub(r'([^a-zA-Z\d\s:]+)', r' \1 ', text)
		    
		    # remove duplicate whitespace
		    #tokenized = re.sub(r' +', ' ', tokenized)

		    # write out tweet text
		    line_text = preprocess(text) + '\n'
		    tweet_text.write(line_text.encode('utf8'))

		    # write out metadata
		    metadata_dict = {k:v for (k,v) in data.iteritems() if k in metadata_list}
		    metadata_dict[u'text'] = line_text.encode('utf8')
		    metadata_dict[u'user_id'] = data[u'user'][u'id']
		    metadata_dict[u'hashtags'] = "|".join(hashtags).encode('utf8')
		    
		    json.dump(metadata_dict, metadata)
		    metadata.write('\n')
