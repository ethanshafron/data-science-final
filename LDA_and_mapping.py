#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:56:44 2018

@author: ethanshafron
"""
### Spatial data and data frame manipulation/plotting
import geopandas as gpd
from ProcessingTweets import TweetsToShapefile, clean
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns

### Text processing
import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy

### LDA Modeling
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim
rcParams.update({'figure.autolayout': True})

""" 
1. get tweets
2. filter only geo tagged tweets
3. filter only tweets in the US
4. Topic model
5. Assign topics to tweets
6. Export as shapefile
7. Assign raster values to points in ArcGIS (Landcover and % imperviousness (bilinear interpolation))
8. Delete points outside of coterminous US
9. Spatially join labeled tweets to county data
10. Calculate proportion of topic by county/state
11. Export as points
12. Create web app using ESRI products
13. Publish
14. Yule Coefficients

"""
###############################################################################
### Set constants, including coordinate reference systems of interest

# North America mapping extents (gonna be relevant when we want to focus on the US)
map_width_m = 5000 * 1000
map_height_m = 3500 * 1000

# WGS84 - the standard geographic coordinate systems
WGS84 = {'init' :'epsg:4326'}

# Albers Equal area - my favorite CRS for North America
albers_usa = {'datum':'NAD83',
              'ellps':'GRS80',
              'proj':'aea', 
              'lat_1':33, 
              'lat_2':45, 
              'lon_0':-97, 
              'lat_0':39, 
              'x_0':map_width_m/2, 
              'y_0':map_height_m/2,
              'units':'m'}

################################## GEO STUFF ###################################

### Read in data as a geodataframe
try:
    GeoOnlyDF = gpd.read_file('Data/GeoOnly/GeoOnly.shp', encoding = 'utf-8')
except:
    TweetsToShapefile('Data/raw-tweets.json', 'GeoOnly', WGS84)
    GeoOnlyDF = gpd.read_file('Data/GeoOnly/GeoOnly.shp', encoding = 'utf-8')


### Check out the spatial distribution of tweets - woah!
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.crs = WGS84
base = world.plot(color='white', edgecolor='black')
GeoOnlyDF.plot(ax=base, color='red', markersize=1)
plt.savefig('spatial_distribution_tweets.png')

### Alright, now we want to get the data into a coordinate system better suited
### for our (my) analysis. This means a projected, as opposed to geographic coordinate system.
### I'm gonna stick with only looking at the US, so Albers equal area is a good choice.
### To do this, I'm gonna create new x and y variables that represent the coordinates
### in meters, the measurement of distance used by this CRS (as opposed to decimal degrees).

USA = world[world.name == 'United States']['geometry'].iloc[0]
USA_mask = GeoOnlyDF.within(USA)
GeoOnlyDF = GeoOnlyDF[USA_mask]
GeoOnlyDF.crs = WGS84
base = world[world.name == 'United States'].plot(color='white', edgecolor='black')
GeoOnlyDF.plot(ax=base, color='red', markersize=1)
plt.savefig('spatial_distribution_US.png')

################# Latent Dicrichlet Allocation Topic Modeling ##################

lemma = WordNetLemmatizer()
nlp = spacy.load('en', disable=['parser', 'ner'])
stop_words = stopwords.words('english')
stop_words.extend(['thankful', 'grateful','come', 'thanksgiving', 'go', 'happy', 'family', 'amp'])

def Tokenize(L):
    for tweet in L:
        yield(gensim.utils.simple_preprocess(str(tweet)))

def lemmatize(l, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for tweet in l:
        doc = nlp(" ".join(tweet)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def remove_stops(l):
    tweets = [[word for word in simple_preprocess(str(tweet)) if word not in stop_words] for tweet in l]
    return tweets

### Take tweets from geodataframe and turn them into a list
Tweets = clean(GeoOnlyDF.text.values.tolist())

### Time to tokenize the words. Gensim has some useful tools for this 
### (see 'Tokenize' function)
TokenizedTweets = list(Tokenize(Tweets))
StopsGone = remove_stops(TokenizedTweets)

### Ok now that we have our tokenized documents, time to create a bigram model
bigram = gensim.models.Phrases(StopsGone, min_count=5, threshold=100)
bigram_model = gensim.models.phrases.Phraser(bigram)

### Make bigrams
bigrams = [bigram_model[tweet] for tweet in StopsGone]

### Lemmatize tokens in each tweet
LemmatizedTweets = lemmatize(bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

### Ok let's get some word counts
id2word = corpora.Dictionary(LemmatizedTweets)
corpus = [id2word.doc2bow(text) for text in LemmatizedTweets]

### And build an LDA model!
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           eval_every=1,
                                           passes=10,
                                           chunksize = 100,
                                           alpha=0.01,
                                           random_state = 50)
lda_model.print_topics()

### Visualize the topic model! 
topics =  pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.show(topics)

########################## Assigning topics to tweets ######################### 

### Ok now we want to assign the most likely topic to each tweet
topics_df = pd.DataFrame()

for doc in corpus:
    
    ### These 2 lines of code took me wayyyyyyyy too long to figure out
    t = lda_model.get_document_topics(doc)
    t = sorted(t, key=lambda x: (x[1]), reverse = True)
  
    ### So now we're iterating through each topic. Let's get the top words for each topic
    for j, (topic_num, prop_topic) in enumerate(t):
        ### Get dominant topic
        words = []
        if j == 0:
            words = lda_model.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in words])
            topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic, 2), 
                                                    topic_keywords]), ignore_index=True)
        else:
            break

topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
contents = pd.Series(LemmatizedTweets)
sent_topics_df = pd.concat([topics_df, contents], axis=1)

### Add topics back to Geodataframe
GeoOnlyDF['label'] = sent_topics_df['Dominant_Topic']
GeoOnlyDF['topic_words'] = sent_topics_df['Topic_Keywords']

### Explore spatial distribution of topics!
USA = world[world.name == 'United States']['geometry'].iloc[0]
USA_mask = GeoOnlyDF.within(USA)
GeoOnlyDF = GeoOnlyDF[USA_mask]
GeoOnlyDF.crs = WGS84
base = world[world.name == 'United States'].plot(color='white', edgecolor='black')

### Plot it!
GeoOnlyDF.plot(ax=base, c=GeoOnlyDF['label'], markersize=5, alpha=.2)
plt.savefig('topic_point_map.png')

GeoOnlyDF.to_file('Data/GeoOnly_US-labeled', driver='ESRI Shapefile', encoding = "utf-8")

### GIS stuff -> same file + spatial variables = GeoOnly_US

################################# Spatial Joining #############################

TweetPointsLabeled = gpd.read_file('Data/GeoOnly_US/GeoOnly_US.shp', encoding = 'utf-8') # more approporiate name
Counties = gpd.read_file('Data/CountyLevelData/CountyLevelData.shp')
Counties = Counties.to_crs(WGS84)
TweetPointsLabeled = TweetPointsLabeled.to_crs(Counties.crs)
CountyTweets = gpd.sjoin(TweetPointsLabeled, Counties, how='left', op='intersects')

### Get counts of each label per county and state, assign dominant topic to county
### Also attach relative abundance of each topic to each tweet for both state and counties
PerCountyTweets = CountyTweets.groupby(['cb_2017_us', 'cb_2017__1', 'label']).size().reset_index(name='count')
PerCountyTotalTweets = CountyTweets.groupby(['cb_2017_us', 'cb_2017__1']).size().reset_index(name='total_county')
CountyProps = pd.merge(PerCountyTweets, PerCountyTotalTweets, left_on= ['cb_2017_us', 'cb_2017__1'], right_on= ['cb_2017_us', 'cb_2017__1'])
CountyProps['proportion_county'] = CountyProps['count']/CountyProps['total_county']

CountyTweets = pd.merge(CountyTweets, CountyProps, left_on= ['cb_2017_us', 'cb_2017__1','label'], right_on= ['cb_2017_us', 'cb_2017__1', 'label'])

PerStateTweets = CountyTweets.groupby(['cb_2017_us', 'label']).size().reset_index(name='count')
PerStateTotalTweets = CountyTweets.groupby('cb_2017_us').size().reset_index(name='total_state')
StateProps = pd.merge(PerStateTweets, PerStateTotalTweets, left_on= 'cb_2017_us', right_on= 'cb_2017_us')
StateProps['proportion_state'] = StateProps['count']/StateProps['total_state']

CountyTweets = pd.merge(CountyTweets, StateProps, left_on= ['cb_2017_us', 'label'], right_on= ['cb_2017_us', 'label'])

### create new data frame, omit text
CountyTweets = CountyTweets.drop(['place'], axis='columns')
CountyTweets = CountyTweets.rename({'PercImpe_4': 'MeanPercentImp', 
                     'topic_word': 'TopicWords',
                     'cb_2017_12': 'MedianHouseInc',
                     'cb_2017_us': 'stateID',
                     'cb_2017__1': 'countyID'}, axis=1)
    
CleanedCountyTweets = CountyTweets[['stateID',
                                    'countyID',
                                    'geometry',
                                    'label',
                                    'TopicWords',
                                    'proportion_state',
                                    'proportion_county',
                                    'Impervious',
                                    'MedianHouseInc',
                                    'LandCover',
                                    'text']]

CleanedCountyTweets['label'] = CleanedCountyTweets['label'].apply(str)
CleanedCountyTweets['LandCover'] = CleanedCountyTweets['LandCover'].apply(str)
CleanedCountyTweets['countyID'] = CleanedCountyTweets['countyID'].apply(str)
CleanedCountyTweets['stateID'] = CleanedCountyTweets['stateID'].apply(str)

CleanedCountyTweets.to_file('Data/CountyTweets', driver='ESRI Shapefile', encoding = "utf-8")

################################ Plotting and EDA #############################
plt.figure()
sns.countplot(x='label', data=CleanedCountyTweets)

plt.figure()
sns.countplot(x='LandCover',  data=CleanedCountyTweets)

plt.figure()
sns.distplot(CleanedCountyTweets['Impervious'], rug=True)

plt.figure()
sns.distplot(CleanedCountyTweets['MedianHouseInc'], rug=True)

plt.figure()
sns.regplot(x='Impervious', y='MedianHouseInc', data=CleanedCountyTweets)

plt.figure()
sns.violinplot(x="LandCover", y='MedianHouseInc', data=CleanedCountyTweets)
plt.xticks(np.arange(15), ('Open Water',
                   'Developed, Open Space',
                   'Developed, Low Intensity',
                   'Developed, Medium Intensity',
                   'Developed, High Intensity',
                   'Barren Land (Rock/Sand/Clay)',
                   'Deciduous Forest',
                   'Evergreen Forest',
                   'Mixed Forest',
                   'Shrub/Scrub',
                   'Grassland/Herbaceous',
                   'Pasture/Hay',
                   'Cultivated Crops',
                   'Woody Wetlands',
                   'Emergent Herbaceous Wetlands'), rotation=90)

plt.figure()
sns.violinplot(x="label", y='MedianHouseInc', data=CleanedCountyTweets)

plt.figure()
sns.violinplot(x="label", y='Impervious', data=CleanedCountyTweets)

### Total tweets per label, faceted by state
plt.figure()
g = sns.FacetGrid(CleanedCountyTweets, col = "stateID", col_wrap = 3)
g = g.map(plt.bar, "label" , "proportion_state")

### Counts of tweets belonging to each topic by land cover class
cross = pd.crosstab(CleanedCountyTweets.LandCover, CleanedCountyTweets.label)
cross.plot.bar(stacked=True)
plt.xticks(np.arange(15), ('Open Water',
                   'Developed, Open Space',
                   'Developed, Low Intensity',
                   'Developed, Medium Intensity',
                   'Developed, High Intensity',
                   'Barren Land (Rock/Sand/Clay)',
                   'Deciduous Forest',
                   'Evergreen Forest',
                   'Mixed Forest',
                   'Shrub/Scrub',
                   'Grassland/Herbaceous',
                   'Pasture/Hay',
                   'Cultivated Crops',
                   'Woody Wetlands',
                   'Emergent Herbaceous Wetlands'))
plt.legend(title='topics')
plt.savefig('land_label_crosstabs.png')

### Percent of tweets belonging to each topic by land cover class
CleanedCountyTweets.groupby(['LandCover', 'label']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.ylabel('Percent of Tweets')
plt.xticks(np.arange(15), ('Open Water',
                   'Developed, Open Space',
                   'Developed, Low Intensity',
                   'Developed, Medium Intensity',
                   'Developed, High Intensity',
                   'Barren Land (Rock/Sand/Clay)',
                   'Deciduous Forest',
                   'Evergreen Forest',
                   'Mixed Forest',
                   'Shrub/Scrub',
                   'Grassland/Herbaceous',
                   'Pasture/Hay',
                   'Cultivated Crops',
                   'Woody Wetlands',
                   'Emergent Herbaceous Wetlands'))
plt.legend(title='topics')
plt.savefig('land_label_percent.png')














    
    
