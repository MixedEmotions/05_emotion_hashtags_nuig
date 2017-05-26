from __future__ import division
import logging
import os
import xml.etree.ElementTree as ET

from senpy.plugins import EmotionPlugin, SenpyPlugin
from senpy.models import Results, EmotionSet, Entry, Emotion

logger = logging.getLogger(__name__)

import numpy as np
import math, itertools
from drevicko.twitter_regexes import cleanString, setupRegexes, tweetPreprocessor
import preprocess_twitter
from collections import defaultdict
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from nltk.tokenize import TweetTokenizer
import nltk.tokenize.casual as casual

import gzip
from datetime import datetime 



class hashTagClassification(EmotionPlugin):
    
    def __init__(self, info, *args, **kwargs):        
        super(hashTagClassification, self).__init__(info, *args, **kwargs)
        self.name = info['name']
        self.id = info['module']
        self._info = info
        hashTagClassification._stop_words = get_stop_words('en')
        local_path = os.path.dirname(os.path.abspath(__file__))  
        
        self._paths_classifiers = os.path.join(os.path.dirname(__file__), 'classifiers')        
        self._paths_word_emb = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'glove.twitter.27B.100d.txt.gz')
        self._paths_word_freq = os.path.join(os.path.dirname(__file__), 'wordFrequencies.dump')
        self._paths_ngramizer = os.path.join(os.path.dirname(__file__), 'ngramizers/ngramizer.dump')
        
        self._emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy'] 
        
        
    def activate(self, *args, **kwargs):

        st = datetime.now()
        self._wordFrequencies = self._load_unique_tokens(filename = self._paths_word_freq)
        logger.info("{} {}".format(datetime.now() - st, "loaded _wordFrequencies"))

        st = datetime.now()
        self._stop_words = get_stop_words('en')
        logger.info("{} {}".format(datetime.now() - st, "loaded _stop_words"))

        st = datetime.now()
        self._ngramizer = joblib.load(self._paths_ngramizer)    
        logger.info("{} {}".format(datetime.now() - st, "loaded _ngramizer"))
        
        #st = datetime.now()
        self._classifiers = {estimator: self._load_classifier(PATH=self._paths_classifiers, ESTIMATOR=estimator) for estimator in self.estimators_list}  
        #logger.info("{} {}".format(datetime.now() - st, "loaded _classifiers"))
        
        st = datetime.now()
        self._Dictionary = self._load_word_vectors(filename = self._paths_word_emb, zipped = True, wordFrequencies = None)
        logger.info("{} {}".format(datetime.now() - st, "loaded _Dictionary"))

        logger.info(self.name+" plugin is ready to go!")
        
        
    def deactivate(self, *args, **kwargs):
        try:
            logger.info(self.name+" plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of "+self.name)

    # CUSTOM FUNCTIONS
    
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
        
        ngramVector = self._tweetToNgramVector(text)
        embeddingsVector = self._ModWordVectors(self._tweetToWordVectors(Dictionary,text))
        additionalVector = self._capitalRatio(text_input)
        
        X = np.asarray( self._bind_vectors((ngramVector,additionalVector, embeddingsVector)) ).reshape(1,-1)   
        return X
        
    def _load_word_vectors(self,  filename = 'glove.twitter.27B.100d.txt.gz', sep = ' ', wordFrequencies = None, zipped = False):
        
        def __read_file(f):
            Dictionary = {}
            for line in f: 
                line_d = line.decode('utf-8').split(sep)
                token = line_d[0]
                token_vector = np.array(line_d[1:], dtype = 'float32')   
                if wordFrequencies:
                    if token in wordFrequencies:                
                        Dictionary[token] = token_vector
                else:
                    Dictionary[token] = token_vector
            return Dictionary
        
        if zipped:
            with gzip.open(filename, 'rb') as f:
                return __read_file(f)
        else:
            with open(filename, 'rb') as f:
                return __read_file(f)


    def _tweetToNgramVector(self, text):        
        return self._ngramizer.transform([text,text]).toarray()[0]       

    def _tweetToWordVectors(self, Dictionary, tweet, fixedLength=False):
        output = []    
        if fixedLength:
            for i in range(100):
                output.append(blankVector)
            for i,token in enumerate(tweet.split()):
                if token in Dictionary:
                    output[i] = Dictionary[token]                
        else:
             for i,token in enumerate(tweet.lower().split()):
                if token in Dictionary:
                    output.append(Dictionary[token])            
        return output
    
    def _ModWordVectors(self, x, mod=True):
        if len(x) == 0:       
            if mod:
                return np.zeros(self.EMBEDDINGS_DIM*3, dtype='float32')
            else:
                return np.zeros(self.EMBEDDINGS_DIM, dtype='float32')
        m = np.matrix(x)
        if mod:
            xMean = np.array(m.mean(0))[0]
            xMin = np.array(m.min(0))[0]
            xMax = np.array(m.max(0))[0]
            xX = np.concatenate((xMean,xMin,xMax))
            return xX
        else:
            return np.array(m.mean(0))[0]
        
    def _bindTwoVectors(self, x0, x1):
        return np.array(list(itertools.chain(x0,x1)),dtype='float32') 
    
    def _bind_vectors(self, x):
        return np.concatenate(x)  
    
    def _capitalRatio(self, tweet):    
        firstCap, allCap = 0, 0
        length = len(tweet)
        if length == 0:
            return np.array([0,0])
        for i,token in enumerate(tweet.split()):
            if( token.istitle() ):
                firstCap += 1
            elif( token.isupper() ):
                allCap += 1
        return np.asarray([firstCap/length,allCap/length])      

    
    def _load_classifier(self, PATH, ESTIMATOR):
        
        models = []
        st = datetime.now()

        for EMOTION in self._emoNames:
            filename = os.path.join(PATH, ESTIMATOR, EMOTION + self.extension_classifier)
            st = datetime.now()
            m = joblib.load(filename)
            logger.info("{} loaded _{}.{}".format(datetime.now() - st, ESTIMATOR, EMOTION))
            models.append( m )
            
        return models
    
    
    def _load_unique_tokens(self, filename = 'wordFrequencies.dump'):    
        return joblib.load(filename)
        

    def _extract_features(self, X, classifiers, estimator):
        if(estimator == 'SVC'):        
            feature_set = {emo: int((clf.predict_proba(X)[0][1])*100) for emo,clf in zip(self._emoNames, classifiers[estimator])}
        else:
            feature_set = {emo: int(clf.predict(X)*100) for emo,clf in zip(self._emoNames, classifiers[estimator])} 
        return feature_set         
    
    
    def analyse(self, **params):
        
        logger.debug("Hashtag SVM Analysing with params {}".format(params))
                
        text_input = params.get("input", None)
        self.ESTIMATOR = params.get("estimator", 'LinearSVC')
        
        
        # EXTRACTING FEATURES
        
        text = self._text_preprocessor(text_input)      
        X = self._convert_text_to_vector(text=text, text_input=text_input, Dictionary=self._Dictionary)   
        feature_text = self._extract_features(X=X, classifiers=self._classifiers, estimator=self.ESTIMATOR)              
            
            
        # GENERATING RESPONSE
        
        response = Results()
        entry = Entry()
        entry.nif__isString = text_input

        emotionSet = EmotionSet()
        emotionSet.id = "Emotions"
        
        if self.ESTIMATOR == 'SVC':
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
            if(self.ESTIMATOR == 'SVC'):
                emotionSet.onyx__hasEmotion.append(Emotion(
                                    onyx__hasEmotionCategory=self.wnaffect_mappings[i],
                                    onyx__hasEmotionIntensity=feature_text[i]))
            else:
                if(feature_text[i] > 0):
                    emotionSet.onyx__hasEmotion.append(Emotion(
                            onyx__hasEmotionCategory=self.wnaffect_mappings[i]))
        
        entry.emotions = [emotionSet,]        
        response.entries.append(entry)
            
        return response
