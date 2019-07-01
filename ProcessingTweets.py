#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:31:49 2018

@author: ethanshafron
"""
import json
import re
import string
import pyLDAvis.gensim
import pandas as pd
import geopandas
from shapely.geometry import Point

def ReadTweetJSON(j):
    tweetlist = []
    with open(j, 'rb') as f:
        for line in f:
            try:
                tweet = json.loads(line)
                tweetlist.append(tweet)
            except Exception as e:
                print(e)
                continue
    return tweetlist
                

def extract_text_from_tweetlist(tweetlist):
    '''Returns list of text from tweets, 
    given a list of tweet dicts'''
    textlist = []
    for tweet in tweetlist:
        try:
            text = tweet['extended_tweet']['full_text']
            textlist.append(text)
        except:
            text = tweet["text"]
            textlist.append(text)
    return textlist

def extract_place_from_tweetlist(tweetlist):
    """
    Returns list of place objects from tweets, 
    given a list of tweet dicts
    """
    placelist = []
    for tweet in tweetlist:
        try:
            place = tweet["place"]['full_name']
            placelist.append(place)
        except TypeError:
            placelist.append(None)
    return placelist

def extract_geo_from_tweetlist(tweetlist):
    """
    Returns list of longitudes from tweets, 
    given a list of tweet dicts
    """
    geolist = []
    for tweet in tweetlist:
        try:
            geo = tweet["geo"]['coordinates']
            geolist.append(geo)
        except TypeError:
            geolist.append(None)
    return geolist

def clean(L):
    ''' 
    Cleans up tweets in a corpus of tweets.
    Removes urls, mentions, and punctuation, makes everything lowercase.
    '''
    lower = [i.lower() for i in L]
    url_free = [re.sub(r'(http?)([\w\-]+)', '', i) for i in lower]
    ment_free = [re.sub(r'(@)([\w\-]+)\b', '', i) for i in url_free]
    punc_free = [''.join(i for i in tweet if i not in string.punctuation) for tweet in ment_free]
    
    return punc_free

def TweetsToShapefile(infile, outfile, CRS):
    
    """ 
    
    Takes in a file of raw tweets and spits out a shapefile of the geolocated ones.
    
    Args:
        infile (str): name of raw tweets in .json format.
        outfile (str): directory/name of shapefile output
        CRS (dict): Coordinate reference system. See geopandas documentation for formatting
        
    Remember - Since the CRS is defined as an argument, there will
    be projection files as well as the shape/database/topology files.
    
    """
    
    ### Read in tweets   
    Tweets = ReadTweetJSON(infile)
    
    ### Create dataframe containing text, coords/geo, label. Add datetime later if needed
    text = extract_text_from_tweetlist(Tweets)
    coords = extract_geo_from_tweetlist(Tweets)
    place = extract_place_from_tweetlist(Tweets)
    label = []
    
    ### Make it a DF
    Thankful = pd.DataFrame(
            {'text':text,
            'label':None,
             'coords':coords,
             'place':place
             })
    
    ### Keep only georeferenced tweets
    GeoOnly = Thankful.dropna(subset=['coords'])
    
    ### For some reason the long/lat are reversed in the tweet metadata. this is dumb.
    for i in GeoOnly['coords']:
        i.reverse()
        
    ### Apply topology to data
    GeoOnly['coords'] = GeoOnly['coords'].apply(Point)
    
    ### Create geodataframe
    GeoOnlyDF = geopandas.GeoDataFrame(GeoOnly, geometry='coords')
    
    GeoOnlyDF.crs = CRS
    
    ### Write to file
    GeoOnlyDF.to_file(outfile, driver='ESRI Shapefile', encoding = "utf-8") 



###############################################################################

