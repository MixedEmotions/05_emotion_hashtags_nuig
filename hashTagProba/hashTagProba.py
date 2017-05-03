
# coding: utf-8

# In[2]:



from __future__ import division
import logging
import os
import xml.etree.ElementTree as ET

from senpy.plugins import EmotionPlugin, SenpyPlugin
from senpy.models import Results, EmotionSet, Entry, Emotion

logger = logging.getLogger(__name__)

# my packages
import codecs, csv, re, nltk
import numpy as np
import math, itertools
from drevicko.twitter_regexes import cleanString, setupRegexes, tweetPreprocessor
import preprocess_twitter
from collections import defaultdict
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.externals import joblib
# from sklearn.svm import SVC, SVR

from nltk.tokenize import TweetTokenizer
import nltk.tokenize.casual as casual

import gzip
from datetime import datetime 

os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model, model_from_json

class hashTagProba(EmotionPlugin):
    
    def __init__(self, info, *args, **kwargs):
        super(hashTagProba, self).__init__(info, *args, **kwargs)
        self.name = info['name']
        self.id = info['module']
        self._info = info
        local_path = os.path.dirname(os.path.abspath(__file__))
        self._categories = {'sadness':[],
                            'disgust':[],
                            'surprise':[],
                            'anger':[],
                            'fear':[],
                            'joy':[]}   

        self._wnaffect_mappings = {'sadness':'sadness',
                                    'disgust':'disgust',
                                    'surprise':'surprise',
                                    'anger':'anger',
                                    'fear':'fear',
                                    'joy':'joy'}
        self._maxlen = 65
        
        self._paths = {
            "word_emb": "glove.twitter.27B.100d.txt",
            "word_freq": 'wordFrequencies.dump',
            "classifiers" : 'classifiers',            
            "ngramizers": 'ngramizers'
            }
        
        self._savedModelPath = local_path + "/classifiers/LSTM/hashTagProba"
        self._path_wordembeddings = os.path.dirname(local_path) + '/glove.twitter.27B.100d.txt.gz'
        
        self._emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy'] 
#         self._emoNames = ['anger','fear','joy','sadness'] 
        
        self._classifiers = {}

        self._Dictionary = {}
        
        self.centroids= {
                            "anger": {
                                "A": 6.95, 
                                "D": 5.1, 
                                "V": 2.7}, 
                            "disgust": {
                                "A": 5.3, 
                                "D": 8.05, 
                                "V": 2.7}, 
                            "fear": {
                                "A": 6.5, 
                                "D": 3.6, 
                                "V": 3.2}, 
                            "joy": {
                                "A": 7.22, 
                                "D": 6.28, 
                                "V": 8.6}, 
                            "sadness": {
                                "A": 5.21, 
                                "D": 2.82, 
                                "V": 2.21},
                            "neutral": {
                                "A": 5.0, 
                                "D": 5.0, 
                                "V": 5.0
                            }
                        }        
        self.emotions_ontology = {
            "anger": "http://gsi.dit.upm.es/ontologies/wnaffect/ns#anger", 
            "disgust": "http://gsi.dit.upm.es/ontologies/wnaffect/ns#disgust", 
            "fear": "http://gsi.dit.upm.es/ontologies/wnaffect/ns#negative-fear", 
            "joy": "http://gsi.dit.upm.es/ontologies/wnaffect/ns#joy", 
            "neutral": "http://gsi.dit.upm.es/ontologies/wnaffect/ns#neutral-emotion",             
            "sadness": "http://gsi.dit.upm.es/ontologies/wnaffect/ns#sadness"
            }
        
        self._centroid_mappings = {
            "V": "http://www.gsi.dit.upm.es/ontologies/onyx/vocabularies/anew/ns#valence",
            "A": "http://www.gsi.dit.upm.es/ontologies/onyx/vocabularies/anew/ns#arousal",
            "D": "http://www.gsi.dit.upm.es/ontologies/onyx/vocabularies/anew/ns#dominance"          
            }
        self._blank = {
            0:[1,0,0,0,0,0],
            1:[0,1,0,0,0,0],
            2:[0,0,1,0,0,0],
            3:[0,0,0,1,0,0],
            4:[0,0,0,0,1,0],
            5:[0,0,0,0,0,1]
        }
        

    def activate(self, *args, **kwargs):
        
        np.random.seed(1337)
        
        st = datetime.now()
        self._hashTagDLModel = self._load_model_and_weights(self._savedModelPath)  
        logger.info("{} {}".format(datetime.now() - st, "loaded _hashTagDLModel"))
        
        st = datetime.now()
        self._Dictionary, self._Indices = self._load_original_vectors(
            filename = self._path_wordembeddings, 
            sep = ' ',
            wordFrequencies = None, 
            zipped = True) # leave wordFrequencies=None for loading the entire WE file
        logger.info("{} {}".format(datetime.now() - st, "loaded _wordEmbeddings"))
        
        logger.info("hashTagProba plugin is ready to go!")
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info("hashTagProba plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of hashTagProba")

    #MY FUNCTIONS
    
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
            return(Dictionary, Indices)
            
        if zipped:
            with gzip.open(filename, 'rb') as f:
                return(__read_file(f))
        else:
            with open(filename, 'rb') as f:
                return(__read_file(f))
            

    def _extract_features(self, X):
        if self._ESTIMATION == 'Probabilities':            
            y_predict = np.array(self._hashTagDLModel.predict(X))[0]
            
        else:
            y_predict = np.array([self._blank[y_] for y_ in self._hashTagDLModel.predict_classes(X)][0])
        feature_set = {emo: y_ for emo, y_ in zip(self._emoNames, y_predict)}
            
        return feature_set       
    
    
    def analyse(self, **params):
        logger.debug("Hashtag LSTM Analysing with params {}".format(params))          
        
        st = datetime.now()
           
        text_input = params.get("input", None)
        
        self._ESTIMATION = params.get("estimation", 'Probabilities')
        text = self._text_preprocessor(text_input)    
        
        X = self._lists_to_vectors(text = text)   
        
        feature_text = self._extract_features(X = X)    
        
            
        response = Results()
       
        entry = Entry()
        entry.nif__isString = text_input
        
        emotionSet = EmotionSet()
        emotionSet.id = "Emotions"
        
        emotion1 = Emotion() 
        for dimension in ['V','A','D']:
            weights = [feature_text[i] for i in feature_text if (i != 'surprise')]
            if not all(v == 0 for v in weights):
                value = np.average([self.centroids[i][dimension] for i in feature_text if (i != 'surprise')], weights=weights) 
            else:
                value = 5.0
            emotion1[self._centroid_mappings[dimension]] = value         

        emotionSet.onyx__hasEmotion.append(emotion1)    
        
        for i in feature_text:
            if self._ESTIMATION == 'Probabilities':
                emotionSet.onyx__hasEmotion.append(Emotion(onyx__hasEmotionCategory=self._wnaffect_mappings[i],
                                    onyx__hasEmotionIntensity=float(feature_text[i])))
            elif self._ESTIMATION == 'Classes':
#                 if(feature_text[i] > 0):
                    emotionSet.onyx__hasEmotion.append(Emotion(onyx__hasEmotionCategory=self._wnaffect_mappings[i],
                                                              onyx__hasEmotionIntensity=int(feature_text[i])))
        
        entry.emotions = [emotionSet,]
        
        response.entries.append(entry)
        
        
        # entry.language = lang
            
        return response

