
# coding: utf-8

# In[121]:



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

import random

os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model, model_from_json

class fivePointRegression(EmotionPlugin):
    
    def __init__(self, info, *args, **kwargs):
        super(fivePointRegression, self).__init__(info, *args, **kwargs)
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
        
        self._vad_mappings = {'confident':'D',
                              'excited':'A',
                              'happy':'V', 
                              'surprised':'S'}
        
        self._maxlen = 65
        
        self._paths = {
            "word_emb": "glove.twitter.27B.100d.txt",
            "word_freq": 'wordFrequencies.dump',
            "classifiers" : 'classifiers',            
            "ngramizers": 'ngramizers'
            }
        
        self._savedModelPath = local_path + "/classifiers/LSTM/fivePointRegression"
        self._path_wordembeddings = os.path.dirname(local_path) + '/glove.twitter.27B.100d.txt.gz'
        
        self._emoNames = ['confident','excited','happy', 'surprised']
#         self._emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy'] 
#         self._emoNames = ['anger','fear','joy','sadness'] 
        
        
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
                                "V": 2.21}
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
            "D": "http://www.gsi.dit.upm.es/ontologies/onyx/vocabularies/anew/ns#dominance",
            "S": "http://www.gsi.dit.upm.es/ontologies/onyx/vocabularies/anew/ns#surprise"
            }
        

    def activate(self, *args, **kwargs):
        
        np.random.seed(1337)
        
        st = datetime.now()
        self._fivePointRegressionModel = self._load_model_and_weights(self._savedModelPath)  
        logger.info("{} {}".format(datetime.now() - st, "loaded _fivePointRegressionModel"))
        
        st = datetime.now()
        self._Dictionary, self._Indices = self._load_original_vectors(
            filename = self._path_wordembeddings, 
            sep = ' ',
            wordFrequencies = None, 
            zipped = True) # leave wordFrequencies=None for loading the entire WE file
        logger.info("{} {}".format(datetime.now() - st, "loaded _wordEmbeddings"))
        
        logger.info("fivePointRegression plugin is ready to go!")
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info("fivePointRegression plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of fivePointRegression")

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
#         if self._ESTIMATION == 'Probabilities':            
#             y_predict = np.array(self._fivePointRegressionModel.predict(X))[0]            
#         else:
#             y_predict = np.array([self._blank[y_] for y_ in self._fivePointRegressionModel.predict_classes(X)][0])
        y_predict = np.array(self._fivePointRegressionModel.predict(X))[0]
        feature_set = {self._vad_mappings[emo]:float(y_) for emo, y_ in zip(self._emoNames, y_predict)}
            
        return feature_set       
    
    # CONVERSION EKMAN TO VAD

    
    def _backwards_conversion(self, original):    
        """Find the closest category"""

        dimensions = list(self.centroids.values())[0]

        def distance(e1, e2):
            return sum((e1[k] - e2.get(k, 0)) for k in dimensions)

        distances = { state:distance(self.centroids[state], original) for state in self.centroids }
        mindistance = max(distances.values())

        for state in distances:
            if distances[state] < mindistance:
                mindistance = distances[state]
                emotion = state

        result = Emotion(onyx__hasEmotionCategory=emotion)
        return result
       
    
    def analyse(self, **params):
        logger.debug("fivePointRegression LSTM Analysing with params {}".format(params))          
        
        st = datetime.now()           
        text_input = params.get("input", None)
        
        text = self._text_preprocessor(text_input)            
        X = self._lists_to_vectors(text = text)           
        feature_text = self._extract_features(X = X)    
        
            
        response = Results()       
        entry = Entry()
        entry.nif__isString = text_input
        
        emotionSet = EmotionSet()
        emotionSet.id = "Emotions"
        
        emotion = Emotion() 
        for dimension in ["V","A","D","S"]:
#             emotion[self._centroid_mappings[dimension]] = float((2+feature_text[dimension])*2.5) 
            emotion[dimension] = float(feature_text[dimension]*10) 
    
        emotionSet.onyx__hasEmotion.append(emotion)  
#         emotionSet.onyx__hasEmotion.append(self._backwards_conversion(emotion))
        
        """
        for semeval
        
        
        
        dimensions = list(self.centroids.values())[0]

        def distance(e1, e2):
            return sum((e1[k] - e2.get(k, 0)) for k in dimensions)

        distances = { state:distance(self.centroids[state], emotion) for state in self.centroids }
        mindistance = max(distances.values())
        
        dummyfix = sorted(distances.values(),reverse=True)

        for state in distances:
            if state != 'joy':
                if distances[state] in dummyfix[0:3]:
                    emotionSet.onyx__hasEmotion.append(
                        Emotion(
                            onyx__hasEmotionCategory = state, 
                            onyx__hasEmotionIntensity = int(1))) 
                else:
                    emotionSet.onyx__hasEmotion.append(
                        Emotion(
                            onyx__hasEmotionCategory = state, 
                            onyx__hasEmotionIntensity = int(0))) 
                
        emotionSet.onyx__hasEmotion.append(
                    Emotion(
                        onyx__hasEmotionCategory = 'surprise', 
                        onyx__hasEmotionIntensity = float((2+feature_text['S'])/4)))
        emotionSet.onyx__hasEmotion.append(
                    Emotion(
                        onyx__hasEmotionCategory = 'joy', 
                        onyx__hasEmotionIntensity = float((2+feature_text['V'])/4)))
        
        emotionSet.prov__wasGeneratedBy = self.id
        
        
        for semeval
        
        """
        
        entry.emotions = [emotionSet,]        
        response.entries.append(entry)
        
        return response


# In[87]:

# centroids= {
#                             "anger": {
#                                 "A": 6.95, 
#                                 "D": 5.1, 
#                                 "V": 2.7}, 
#                             "disgust": {
#                                 "A": 5.3, 
#                                 "D": 8.05, 
#                                 "V": 2.7}, 
#                             "fear": {
#                                 "A": 6.5, 
#                                 "D": 3.6, 
#                                 "V": 3.2}, 
#                             "joy": {
#                                 "A": 7.22, 
#                                 "D": 6.28, 
#                                 "V": 8.6}, 
#                             "sadness": {
#                                 "A": 5.21, 
#                                 "D": 2.82, 
#                                 "V": 2.21}
#                         }   


# In[116]:

# def _backwards_conversion(original):    
#         """Find the closest category"""
        
#         dimensions = list(centroids.values())[0]
        
#         def distance(e1, e2):
#             return sum((e1[k] - e2.get(k, 0)) for k in dimensions)
        
#         def _vectors_similarity(v1 , v2):
#             return( 1 - spatial.distance.cosine(v1,v2) )

#         distances = { state:abs(distance(centroids[state], original)) for state in centroids }
#         print(np.array(centroids['anger'].values()))
#         distances2 = {state:_vectors_similarity(centroids[state].values() , feature_text.values())  for state in centroids}
#         mindistance = max(distances.values())
#         print(distances)
#         print(distances2)
#         for state in distances:
#             if distances[state] < mindistance:
#                 mindistance = distances[state]
#                 emotion = state
                
#         result = Emotion(onyx__hasEmotionCategory=emotion, onyx__hasEmotionIntensity=emotion)
#         return result
    
# feature_text = {
#     "A":5.9574053436517715,
#     "D":6.3352929055690765,
#     "V":2.9072564840316772

# }

# import numpy as np
# from senpy.models import Emotion
# from scipy import spatial

# emotion = Emotion() 
# for dimension in ["V","A","D"]:
#     emotion[dimension] = float((feature_text[dimension])) 
    
# _backwards_conversion(emotion)


# In[115]:

# for state in centroids:
# #     print(centroids[state])
# #     print([i for i in feature_text.values()])
# #     print(([i for i in centroids[state].values()]))
#     print(state)
#     print(_vectors_similarity(
#             [i for i in feature_text.values()],
#             [i for i in centroids[state].values()]))

