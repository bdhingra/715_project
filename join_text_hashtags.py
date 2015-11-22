import json
import sys
import io
import re

MIN_CHAR = 5

textFile = sys.argv[1]
hashFile = sys.argv[2]
outFile = sys.argv[3]

with io.open(textFile,'r',encoding='utf-8') as text, io.open(hashFile,'r',encoding='utf-8') as tag, io.open(outFile,'w',encoding='utf-8') as out:
    tid = 0
    while True:
        try:
            t = text.readline()
            if not t:
                break
            h = json.loads(tag.readline())
        except:
            continue
	
	if len(t) <= MIN_CHAR+1:
		continue

	if h:
		for ht in h:
			hasht = re.sub(r'^(.+?)\1+$', r'\1', ht.lower())
			out.write('%s\t%d\t%s' % (hasht,tid,t))
		tid += 1
