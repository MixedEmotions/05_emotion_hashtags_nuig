# -*- coding: utf-8 -*-

from __future__ import division
import logging
import os
import xml.etree.ElementTree as ET

from senpy.plugins import EmotionPlugin, SenpyPlugin
from senpy.models import Results, EmotionSet, Entry, Emotion

logger = logging.getLogger(__name__)

# my packages
import numpy as np
import math, itertools
from drevicko.twitter_regexes import cleanString, setupRegexes, tweetPreprocessor
from collections import defaultdict
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.svm import SVC, SVR

from nltk.tokenize import TweetTokenizer
import nltk.tokenize.casual as casual




class hashTagClassification(EmotionPlugin):
    
    def __init__(self, info, *args, **kwargs):
        super(hashTagClassification, self).__init__(info, *args, **kwargs)
        self.name = info['name']
        self.id = info['module']
        self._info = info
        hashTagClassification._stop_words = get_stop_words('en')
        local_path=os.path.dirname(os.path.abspath(__file__))
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

        self.EXTENSION = '.dump'
        self._emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy']  
        self.EMBEDDING_DIM = 100  
        self.WORD_FREQUENCY_TRESHOLD = 10
        
        self._uniqueTokens = {}

        self._classifiers = []
        self._DATA_FORMAT = 'weng'
        self._ESTIMATOR = 'LinearSVC' 
        self._Dictionary = {}


    def activate(self, *args, **kwargs):
        
        self._Dictionary = self._load_word_vectors()
        self._uniqueTokens = self._load_unique_tokens(filename = 'uniqueTokens.dump')
        self._classifiers = self._load_classifiers(DATA_FORMAT=self._DATA_FORMAT, ESTIMATOR=self._ESTIMATOR, emoNames=self._emoNames)
        self._stop_words = get_stop_words('en')

        # SEP = '/'
        self._ngramizers = []                              
        for n_grams in [2,3,4]:
            filename = os.path.join(os.path.dirname(__file__), 'LinearSVC/', str(n_grams) + 'gramizer' + self.EXTENSION)
            #filename = 'LinearSVR' +SEP+ 'ngramizer'+str(n_grams) + self.EXTENSION
            #filename = os.path.join(os.path.dirname(__file__),filename)
            self._ngramizers.append( joblib.load(filename) )

        logger.info("hashTagClassification plugin is ready to go!")
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info("hashTagClassification plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of hashTagClassification")

    #MY FUNCTIONS
    
    def _text_preprocessor(self, text):
        
        text = text.replace("'","") # this is for contractions eg: "don't --> dont" instead of "don't --> don 't"
        text = casual._replace_html_entities(text) 
        text = casual.remove_handles(text)
        text = casual.reduce_lengthening(text)
        text = cleanString(setupRegexes('twitterProAna'),text)                

        text = ' '.join([span for notentity,span in tweetPreprocessor(text, ("urls", "users", "lists")) if notentity])        
        return text
    
    def _convert_text_to_vector(self, text, Dictionary, DATA_FORMAT):
              
        tmp = []
        for token in text.split():
            try:
                if(self._uniqueTokens[token] >= self.WORD_FREQUENCY_TRESHOLD):
                    if(not token in self._stop_words):
                        tmp.append(token)
            except IndexError:
                pass
        text = ' '.join(tmp)
             
        X = []        
        
        n2gramVector,n3gramVector,n4gramVector = self._tweetToNgramVector(text)
        embeddingsVector = self._ModWordVectors(self._tweetToWordVectors(Dictionary,text))
        additionalVector = self._capitalRatio(text)
        
        X4 = np.asarray(self._bindTwoVectors(n4gramVector,self._bindTwoVectors(additionalVector, embeddingsVector)) ).reshape(1,-1) 
        X3 = np.asarray(self._bindTwoVectors(n3gramVector,self._bindTwoVectors(additionalVector, embeddingsVector)) ).reshape(1,-1)
        X2 = np.asarray(self._bindTwoVectors(n2gramVector,self._bindTwoVectors(additionalVector, embeddingsVector)) ).reshape(1,-1)

        X = {'sadness':X4,'disgust':X4,'surprise':X4, 'anger':X2, 'fear':X4,'joy':X3}

        return(X)

    def _load_word_vectors(self,  filename="wordvectors-glove.twitter.27B.100d"):
        
        Dictionary = {}
        for line in open(os.path.join(os.path.dirname(__file__),filename), 'rb'): 
            line_d = line.decode('utf-8').split(', ')
            token, token_id = line_d[0], line_d[1]
            token_vector = np.array(line_d[2:], dtype = 'float32')    
            Dictionary[token] = token_vector

        return(Dictionary)
    
    def _tweetToNgramVector(self, text):
        return(self._ngramizers[0].transform([text,text]).toarray()[0] , self._ngramizers[1].transform([text,text]).toarray()[0], self._ngramizers[2].transform([text,text]).toarray()[0])        

    def _tweetToWordVectors(self, Dictionary, tweet, fixedLength=False):
        output = []    
        if(fixedLength):
            for i in range(100):
                output.append(blankVector)
            for i,token in enumerate(tweet.split()):
                if token in Dictionary:
                    output[i] = Dictionary[token]                
        else:
             for i,token in enumerate(tweet.lower().split()):
                if token in Dictionary:
                    output.append(Dictionary[token])            
        return(output)
    def _ModWordVectors(self, x, mod=True):
        if(len(x) == 0):       
            if(mod):
                return(np.zeros(self.EMBEDDING_DIM*3, dtype='float32'))
            else:
                return(np.zeros(self.EMBEDDING_DIM, dtype='float32'))
        m =  np.matrix(x)
        if(mod):
            xMean = np.array(m.mean(0))[0]
            xMin = np.array(m.min(0))[0]
            xMax = np.array(m.max(0))[0]
            xX = np.concatenate((xMean,xMin,xMax))
            return(xX)
        else:
            return(np.array(m.mean(0))[0])
    def _bindTwoVectors(self, x0,x1):
        xX = np.array(list(itertools.chain(x0,x1)),dtype='float32')
        return(xX) 
    
    def _capitalRatio(self, tweet):
    
        firstCap, allCap = 0, 0
        length = len(tweet)
        if length==0:
            return np.array([0,0])

        for i,token in enumerate(tweet.split()):
            if( token.istitle() ):
                firstCap += 1
            elif( token.isupper() ):
                allCap += 1
        return(np.asarray([firstCap/length,allCap/length]))      

    
    def _load_classifiers(self, DATA_FORMAT, ESTIMATOR, emoNames):
        
        SEP = '/'
        models = []
                
        for EMOTION in range(len(emoNames)):
            filename = ESTIMATOR +SEP+ DATA_FORMAT +SEP+ str(emoNames[EMOTION]) + self.EXTENSION
            filename = os.path.join(os.path.dirname(__file__),filename)
            m = joblib.load(filename)
            # print(m)
            models.append( m )
            
            #print(filename + ' loaded')
        return(models)
    
    
    def _load_unique_tokens(self, filename = 'uniqueTokens.dump'):
    
        filename = os.path.join(os.path.dirname(__file__),filename)
        return(joblib.load(filename))
        

    def _compare_tweets(self, X, classifiers):
        
        #feature_set = {emo: int(clf.predict(X)) for emo,clf in zipself._(self.emoNames, classifiers)} 
        feature_set = {emo: int(clf.predict(X[emo])) for emo,clf in zip(self._emoNames, classifiers)} 
        return feature_set        
    
    
    def analyse(self, **params):
        logger.debug("Hashtag SVM Analysing with params {}".format(params))
        
        text_input = params.get("input", None) 
        text = self._text_preprocessor(text_input)        
        X = self._convert_text_to_vector(text, self._Dictionary, self._DATA_FORMAT) 
        feature_text = self._compare_tweets(X=X, classifiers=self._classifiers)
        response = Results()

        entry = Entry(id="Entry",
                      text=text_input)
        emotionSet = EmotionSet(id="Emotions0")
        emotions = emotionSet.onyx__hasEmotion

        for i in feature_text:
            emotions.append(Emotion(onyx__hasEmotionCategory=self._wnaffect_mappings[i],
                                    onyx__hasEmotionIntensity=feature_text[i]))

        entry.emotions = [emotionSet]
        response.entries.append(entry)
        return response
