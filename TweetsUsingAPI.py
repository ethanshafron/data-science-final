#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:41:23 2018

@author: ethanshafron
"""

import tweepy
from TwitterAuth import consumer_key, consumer_secret, access_token, access_secret
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json



### Authenticate App and gain access to APIs
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth, 
                 wait_on_rate_limit=True, 
                 wait_on_rate_limit_notify=True)


### Connect to the streaming API and modify the listener class
### so that it only pulls original tweets and saves them to a json file
class MyListener(StreamListener):
 
    def on_data(self, data):
        tweet = json.loads(data)
        try:
            if 'RT @' not in tweet['text']:
                with open('raw-tweets.json', 'a') as f:
                    f.write(data)
                    return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status_code):
        print(status_code)
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False

myStreamListener = MyListener()
twitter_stream = Stream(auth= auth, listener = myStreamListener)

### Get english tweets containing the below phrases
twitter_stream.filter(languages = ["en"], 
                      track=["thankful for", "grateful for", "blessed"])




