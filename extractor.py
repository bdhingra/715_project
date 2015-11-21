# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:47:41 2015

@author: zhong
"""
'''
def extract(f_name): 
    with open(f_name) as infile:
        result = []
        #temp = 0
        tweets = infile.readlines()
        print(len(tweets))
        print(tweets[0])
    return result    
    
#extract('tweets.2014-09-30T22_42_47')
'''
import gzip
import glob
import os

def process1(): 
    path = os.getcwd() + "/raw_tweets/"
    os.chdir(path)
    for infile in glob.glob(*):
        if infile.endwith('.gz'): 
            with gzip.open('file.txt.gz', 'rb') as f:   
                file_content = f.read()