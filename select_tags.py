import json
import sys

from collections import defaultdict

dictPath = sys.argv[1]
tagPath = sys.argv[2]
outPath = sys.argv[3]

def create_dict(f):
	tag_dict = defaultdict(int)
	for line in f:
		row = line.rstrip('\n').split(' - ')
		tag_dict[row[0]] = int(row[1])
	return tag_dict

def add_tags(f_in, f_out, tag_dict):
	for line in f_in:
		current = json.loads(line)
		if current[0] in tag_dict:
			f_out.write(line)

def remove_tags(f_in, f_out, tag_dict):
	for line in f_in:
		current = json.loads(line)
		if not current[0] in tag_dict:
			f_out.write(line)

tag_dict = create_dict(open(tagPath,'r'))
remove_tags(open(dictPath,'r'), open(outPath,'w'), tag_dict)
print sum(tag_dict.values())
