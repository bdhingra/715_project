from guineapig import *
import json

#parse params
params = GPig.getArgvParams()
in_file= params['input']
MIN_T = int(params['mint'])

# main
class HashDict(Planner):
    hashdict = ReadLines(in_file) | ReplaceEach(by=lambda line:line.rstrip('\n').split('\t')) | Group(by=lambda (h,tid,t):h, retaining=lambda (h,tid,t):(tid,t)) | Filter(by=lambda (h,t):len(t)>MIN_T) | Format(by=lambda line:json.dumps(line))

if __name__ == "__main__":
    planner = HashDict()
    planner.main(sys.argv)
