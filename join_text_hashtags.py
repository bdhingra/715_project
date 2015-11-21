import json
import sys
import io

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

        if not h:
            out.write('\t%d\t%s' % (tid,t))
        else:
            for ht in h:
                out.write('%s\t%d\t%s' % (ht,tid,t))
        tid += 1
