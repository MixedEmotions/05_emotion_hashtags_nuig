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
import preprocess_twitter
from collections import defaultdict
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.svm import SVC, SVR

from nltk.tokenize import TweetTokenizer
import nltk.tokenize.casual as casual

import gzip




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
        self._paths = {
            "word_emb": "glove.twitter.27B.200d.cut.txt",
            "word_freq": 'wordFrequencies.dump',
            "classifiers" : 'classifiers',            
            "ngramizers": 'ngramizers'
            }
        self._emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy']  
        self.EMBEDDINGS_DIM = 200  
        self.WORD_FREQUENCY_TRESHOLD = 3
        
        self._wordFrequencies = {}

        self._classifiers = {}
        self.ESTIMATOR = 'LinearSVC'
        self._estimators_list = ['SVC', 'LinearSVC']
        self._Dictionary = {}
        
        self.centroids= {
                            "anger": {
                                "A": 6.95, 
                                "D": 5.1, 
                                "V": 2.7
                            }, 
                            "disgust": {
                                "A": 5.3, 
                                "D": 8.05, 
                                "V": 2.7
                            }, 
                            "fear": {
                                "A": 6.5, 
                                "D": 3.6, 
                                "V": 3.2
                            }, 
                            "joy": {
                                "A": 7.22, 
                                "D": 6.28, 
                                "V": 8.6
                            }, 
                            "sadness": {
                                "A": 5.21, 
                                "D": 2.82, 
                                "V": 2.21
                            },
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
        

    def activate(self, *args, **kwargs):
        
        
        self._Dictionary = self._load_word_vectors(filename= self._paths["word_emb"], zipped = False)
        self._wordFrequencies = self._load_unique_tokens(filename = self._paths["word_freq"])
        
        self._classifiers = {estimator: self._load_classifier(PATH=self._paths["classifiers"], ESTIMATOR=estimator, emoNames=self._emoNames) for estimator in self._estimators_list}
    
        self._stop_words = get_stop_words('en')

        self._ngramizers = []                              
        for n_grams in [2,3,4]:
            filename = os.path.join(os.path.dirname(__file__), self._paths["ngramizers"], str(n_grams)+'gramizer.dump')
            self._ngramizers.append( joblib.load(filename) )

        logger.info("hashTagClassification plugin is ready to go!")
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info("hashTagClassification plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of hashTagClassification")

    #MY FUNCTIONS
    
    def _text_preprocessor(self, text):
        
        text = preprocess_twitter.tokenize(text)
        
        text = casual.reduce_lengthening(text)
        text = cleanString(setupRegexes('twitterProAna'),text)  
        text = ' '.join([span for notentity,span in tweetPreprocessor(text, ("urls", "users", "lists")) if notentity]) 
        text = text.replace('\t','')
        text = text.replace('< ','<').replace(' >','>')
        text = text.replace('):', '<sadface>').replace('(:', '<smile>')
        return ' '.join(text.split())

    def tokenise_tweet(text):
        text = preprocess_twitter.tokenize(text)
        text = preprocess_tweet(text)     
        return ' '.join(text.split())
    
    def _convert_text_to_vector(self, text, text_input, Dictionary):
              
        tmp = []
        for token in text.split():
            try:
                if(self._wordFrequencies[token] >= self.WORD_FREQUENCY_TRESHOLD):
                    if(not token in self._stop_words):
                        tmp.append(token)
            except IndexError:
                pass
        text = ' '.join(tmp)
             
        X = []        
        
        n2gramVector,n3gramVector,n4gramVector = self._tweetToNgramVector(text)
        embeddingsVector = self._ModWordVectors(self._tweetToWordVectors(Dictionary,text))
        additionalVector = self._capitalRatio(text_input)
        
        X4 = np.asarray( self._bind_vectors((n4gramVector,additionalVector, embeddingsVector)) ).reshape(1,-1) 
        X3 = np.asarray( self._bind_vectors((n3gramVector,additionalVector, embeddingsVector)) ).reshape(1,-1)
        X2 = np.asarray( self._bind_vectors((n2gramVector,additionalVector, embeddingsVector)) ).reshape(1,-1)

        X = {'sadness':X4, 'disgust':X4, 'surprise':X4, 'anger':X2, 'fear':X4, 'joy':X3}

        return(X)
        
    def _load_word_vectors(self,  filename = 'glove.twitter.27B.200d.txt.gz', sep = ' ', wordFrequencies = None, zipped = False):
        
        filename = os.path.join(os.path.dirname(__file__),filename)
        Dictionary = {}
        
        if(zipped): 
            f = gzip.open(filename, 'rb')
        else: 
            f = open(filename, 'rb')
            
        for line in f: 
            line_d = line.decode('utf-8').split(sep)
            token = line_d[0]
            token_vector = np.array(line_d[1:], dtype = 'float32')   
            if(wordFrequencies):
                if(token in wordFrequencies):                
                    Dictionary[token] = token_vector
            else:
                Dictionary[token] = token_vector
        
        f.close()
        
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
                return(np.zeros(self.EMBEDDINGS_DIM*3, dtype='float32'))
            else:
                return(np.zeros(self.EMBEDDINGS_DIM, dtype='float32'))
        m =  np.matrix(x)
        if(mod):
            xMean = np.array(m.mean(0))[0]
            xMin = np.array(m.min(0))[0]
            xMax = np.array(m.max(0))[0]
            xX = np.concatenate((xMean,xMin,xMax))
            return(xX)
        else:
            return(np.array(m.mean(0))[0])
        
    def _bindTwoVectors(self, x0, x1):
        xX = np.array(list(itertools.chain(x0,x1)),dtype='float32')
        return(xX) 
    
    def _bind_vectors(self, x):
        return np.concatenate(x)  
    
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

    
    def _load_classifier(self, PATH, ESTIMATOR, emoNames):
        
        SEP = '/'
        models = []
                
        for EMOTION in range(len(emoNames)):
            filename = PATH+SEP+ESTIMATOR+SEP+ str(emoNames[EMOTION]) + self.EXTENSION
            filename = os.path.join(os.path.dirname(__file__),filename)
            m = joblib.load(filename)
            models.append( m )
            
        return(models)
    
    
    def _load_unique_tokens(self, filename = 'wordFrequencies.dump'):
    
        filename = os.path.join(os.path.dirname(__file__),filename)
        return(joblib.load(filename))
        

    def _extract_features(self, X, classifiers, estimator):
        if(estimator == 'SVC'):        
            feature_set = {emo: int((clf.predict_proba(X[emo])[0][1])*100) for emo,clf in zip(self._emoNames, classifiers[estimator])}
        else:
            feature_set = {emo: int(clf.predict(X[emo])*100) for emo,clf in zip(self._emoNames, classifiers[estimator])} 
            
        return feature_set        
    
    
    def analyse(self, **params):
        logger.debug("Hashtag SVM Analysing with params {}".format(params))
                
        text_input = params.get("input", None)
        self.ESTIMATOR = params.get("estimator", 'LinearSVC')
        
        text = self._text_preprocessor(text_input)        
        
        X = self._convert_text_to_vector(text=text, text_input=text_input, Dictionary=self._Dictionary)   
            
        feature_text = self._extract_features(X=X, classifiers=self._classifiers, estimator=self.ESTIMATOR)              
            
        response = Results()

        entry = Entry()
        entry.nif__isString = text_input

        emotionSet = EmotionSet()
        emotionSet.id = "Emotions"

        emotion1 = Emotion()        

        for dimension in ['V','A','D']:
            value = np.average([self.centroids[i][dimension] for i in feature_text if (i != 'surprise')], weights=[feature_text[i] for i in feature_text if (i != 'surprise')]) 
            emotion1[self._centroid_mappings[dimension]] = value         

        emotionSet.onyx__hasEmotion.append(emotion1)    
        
        for i in feature_text:
            emotionSet.onyx__hasEmotion.append(Emotion(onyx__hasEmotionCategory=self._wnaffect_mappings[i],
                                    onyx__hasEmotionIntensity=feature_text[i]))
        
        entry.emotions = [emotionSet,]
        
        response.entries.append(entry)
        # entry.language = lang
            
        return response
