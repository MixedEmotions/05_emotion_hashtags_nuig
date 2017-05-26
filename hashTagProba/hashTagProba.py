
# coding: utf-8

# In[2]:

from __future__ import division
import logging
import os
os.environ['KERAS_BACKEND'] = 'theano'
import xml.etree.ElementTree as ET

from senpy.plugins import EmotionPlugin, SenpyPlugin
from senpy.models import Results, EmotionSet, Entry, Emotion

logger = logging.getLogger(__name__)

import codecs, csv, re, nltk
import numpy as np
import math, itertools
from drevicko.twitter_regexes import cleanString, setupRegexes, tweetPreprocessor
import preprocess_twitter
from collections import defaultdict
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import TweetTokenizer
import nltk.tokenize.casual as casual

import gzip
from datetime import datetime 


from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model, model_from_json


from sklearn.externals import joblib

class hashTagProba(EmotionPlugin):
    
    def __init__(self, info, *args, **kwargs):
        super(hashTagProba, self).__init__(info, *args, **kwargs)
        self.name = info['name']
        self.id = info['module']
        self._info = info
        local_path = os.path.dirname(os.path.abspath(__file__))
        self.testing = False
        self._categories = {'sadness':[],
                            'disgust':[],
                            'surprise':[],
                            'anger':[],
                            'fear':[],
                            'joy':[]}   
        self._maxlen = 65
                
        self._savedModelPath = local_path + "/classifiers/LSTM/hashTagProba"
        self._path_wordembeddings = os.path.dirname(local_path) + '/glove.twitter.27B.100d.txt.gz'
        
        self.emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy'] 
        
        
    def _load_unique_tokens(self, filename = 'wordFrequencies.dump'):    
        return joblib.load(filename)    
        

    def activate(self, *args, **kwargs):
        
        np.random.seed(1337)
        
        st = datetime.now()
        self._hashTagDLModel = self._load_model_and_weights(self._savedModelPath)  
        logger.info("{} {}".format(datetime.now() - st, "loaded _hashTagDLModel"))
        
        if self.testing:
            st = datetime.now()
            self._wordFrequencies = self._load_unique_tokens(
                filename = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 'hashTagClassification', 'wordFrequencies.dump'))
            logger.info("{} {}".format(datetime.now() - st, "loaded _wordFrequencies"))
        else:
            self._wordFrequencies = None        
        
        st = datetime.now()
        self._Dictionary, self._Indices = self._load_original_vectors(
            filename = self._path_wordembeddings, 
            sep = ' ',
            wordFrequencies = self._wordFrequencies, 
            zipped = True) # leave wordFrequencies=None for loading the entire WE file
        logger.info("{} {}".format(datetime.now() - st, "loaded _wordEmbeddings"))
        
        logger.info("%s plugin is ready to go!" % self.name)
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info("%s plugin is being deactivated..." % self.name)
        except Exception:
            print("Exception in logger while reporting deactivation of %s" % self.name)

    # CUSTOM FUNCTIONS
    
    def _load_model_and_weights(self, filename):
        with open(filename+'.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            
        loaded_model.load_weights(filename+'.h5')
        
        return loaded_model
    
    def _lists_to_vectors(self, text):
        train_sequences = [self._text_to_sequence(text)]  
        X = sequence.pad_sequences(train_sequences, maxlen=self._maxlen)
        return X
    
    def _text_to_sequence(self,text):
        train_sequence = []
        for token in text.split():
            try:
                train_sequence.append(self._Indices[token])
            except:
                train_sequence.append(0)
        train_sequence.extend([0]*( self._maxlen-len(train_sequence)) )
        return np.array(train_sequence)  
    
    def _text_preprocessor(self, text):        
        text = preprocess_twitter.tokenize(text)        
        text = casual.reduce_lengthening(text)
        text = cleanString(setupRegexes('twitterProAna'),text)  
        text = ' '.join([span for notentity,span in tweetPreprocessor(text, ("urls", "users", "lists")) if notentity]) 
        text = text.replace('\t','')
        text = text.replace('< ','<').replace(' >','>')
        text = text.replace('):', '<sadface>').replace('(:', '<smile>')
        text = text.replace(" 't", "t")#.replace("#", "")
        return ' '.join(text.split())

    def tokenise_tweet(text):
        text = preprocess_twitter.tokenize(text)
        text = preprocess_tweet(text)     
        return ' '.join(text.split())
    
    
    def _load_original_vectors(self, filename = 'glove.27B.100d.txt', sep = ' ', wordFrequencies = None, zipped = False):
       
        def __read_file(f):
            Dictionary, Indices  = {},{}
            i = 1
            for line in f:
                line_d = line.decode('utf-8').split(sep)

                token = line_d[0]
                token_vector = np.array(line_d[1:], dtype = 'float32')   
                if(wordFrequencies):
                    if(token in wordFrequencies):                
                        Dictionary[token] = token_vector
                        Indices.update({token:i})
                        i+=1
                else:
                    Dictionary[token] = token_vector
                    Indices.update({token:i})
                    i+=1
            return Dictionary, Indices
            
        if zipped:
            with gzip.open(filename, 'rb') as f:
                return __read_file(f)
        else:
            with open(filename, 'rb') as f:
                return __read_file(f)
            

    def _extract_features(self, X):
        
        if self._ESTIMATION == 'Probabilities':            
            y_predict = np.array(self._hashTagDLModel.predict(X))[0]            
        else:
            blank = [0] * len(self.emoNames)
            for i,pred in enumerate(self._hashTagDLModel.predict_classes(X)):
                blank[pred] = 1
            y_predict = np.array(blank)
            
        feature_set = {emo: y_ for emo, y_ in zip(self.emoNames, y_predict)}
            
        return feature_set       
    
    
    def analyse(self, **params):
        
        logger.debug("Hashtag LSTM Analysing with params {}".format(params))          
                  
        text_input = params.get("input", None)        
        self._ESTIMATION = params.get("estimation", 'Probabilities')
        
        
        # EXTRACTING FEATURES
        
        text = self._text_preprocessor(text_input)    
        
        X = self._lists_to_vectors(text = text)   
        feature_text = self._extract_features(X = X)    
        
        
        # GENERATING RESPONSE   
        
        response = Results()
       
        entry = Entry()
        entry.nif__isString = text_input
        
        emotionSet = EmotionSet()
        emotionSet.id = "Emotions"
        
        if self._ESTIMATION == 'Probabilities':
            emotionSet.onyx__maxIntensityValue = float(100.0)
        
        emotion1 = Emotion() 
        for dimension in ['V','A','D']:
            weights = [feature_text[i] for i in feature_text if (i != 'surprise')]
            if not all(v == 0 for v in weights):
                value = np.average([self.centroids[i][dimension] for i in feature_text if (i != 'surprise')], weights=weights) 
            else:
                value = 5.0
            emotion1[self.centroid_mappings[dimension]] = value         

        emotionSet.onyx__hasEmotion.append(emotion1)    
        
        for i in feature_text:
            if self._ESTIMATION == 'Probabilities':
                emotionSet.onyx__hasEmotion.append(Emotion(
                        onyx__hasEmotionCategory=self.wnaffect_mappings[i],
                        onyx__hasEmotionIntensity=float(feature_text[i])*100 ))
            elif self._ESTIMATION == 'Classes':
                if feature_text[i] > 0:
                    emotionSet.onyx__hasEmotion.append(Emotion(
                        onyx__hasEmotionCategory = self.wnaffect_mappings[i]))                    
                        #onyx__hasEmotionIntensity=int(feature_text[i])))
        
        entry.emotions = [emotionSet,]        
        response.entries.append(entry)
            
        return response

