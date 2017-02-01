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
        self.id = info['module']
        self.info = info
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
        self.emoNames = ['sadness', 'disgust', 'surprise', 'anger', 'fear', 'joy']  
        self.EMBEDDING_DIM = 100  
        self.WORD_FREQUENCY_TRESHOLD = 10
        
        self._uniqueTokens = {}


    def activate(self, *args, **kwargs):
        print("about to load dictionary")
        self._uniqueTokens = self._load_unique_tokens(filename = 'uniqueTokens.dump')
        self._Dictionary = self._load_word_vectors()
        self._stop_words = get_stop_words('en')
        logger.info("EmoText plugin is ready to go!")

    def deactivate(self, *args, **kwargs):
        try:
            logger.info("EmoText plugin is being deactivated...")
        except Exception:
            print("Exception in logger while reporting deactivation of hashTagClassification")

    #MY FUNCTIONS
    
    def _text_preprocessor(self, text):
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
        
        if(DATA_FORMAT == 'we'):
            embeddingsVector = self.ModWordVectors(self.tweetToWordVectors(Dictionary,text))
            additionalVector = self.capitalRatio(text)
            Xa = self.bindTwoVectors(additionalVector, embeddingsVector)  
            X = {'sadness':Xa,'disgust':Xa,'surprise':Xa,'anger':Xa,'fear':Xa,'joy':Xa}
                
        elif(DATA_FORMAT == 'ng'):            
            bigramVector,trigramVector = self.tweetToNgramVector(text)
            additionalVector = self.capitalRatio(text)
            Xa = self.bindTwoVectors(bigramVector,additionalVector) 
            X = {'sadness':Xa,'disgust':Xa,'surprise':Xa,'anger':Xa,'fear':Xa,'joy':Xa}
                
        elif(DATA_FORMAT == 'weng'):
            bigramVector,trigramVector = self.tweetToNgramVector(text)
            embeddingsVector = self.ModWordVectors(self.tweetToWordVectors(Dictionary,text))
            additionalVector = self.capitalRatio(text)
            Xa = np.asarray(self.bindTwoVectors(bigramVector,self.bindTwoVectors(additionalVector, embeddingsVector)) ) 
            Xb = np.asarray(self.bindTwoVectors(trigramVector,self.bindTwoVectors(additionalVector, embeddingsVector)) ) 
            
            X = {'sadness':Xa,'disgust':Xa,'surprise':Xb,'anger':Xa,'fear':Xb,'joy':Xa}

        return(X)

    def _load_word_vectors(self,  filename="wordvectors-glove.twitter.27B.100d"):
        
        Dictionary = {}
        for line in open(os.path.join(os.path.dirname(__file__),filename), 'rb'): 
            line_d = line.decode('utf-8').split(', ')
            token, token_id = line_d[0], line_d[1]
            token_vector = np.array(line_d[2:], dtype = 'float32')    
            Dictionary[token] = token_vector

        return(Dictionary)
    
    def tweetToNgramVector(self, text):
        SEP = '/'
        ngramizers = []
                              
        for n_grams in [2,3]:
            filename = os.path.join(os.path.dirname(__file__), 'LinearSVC/', 'ngramizer' + str(n_grams) + self.EXTENSION)
            #filename = 'LinearSVR' +SEP+ 'ngramizer'+str(n_grams) + self.EXTENSION
            #filename = os.path.join(os.path.dirname(__file__),filename)
            ngramizers.append( joblib.load(filename) )
        
        return(ngramizers[0].transform([text,text]).toarray()[0] , ngramizers[1].transform([text,text]).toarray()[0])        

    def tweetToWordVectors(self, Dictionary, tweet, fixedLength=False):
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
    def ModWordVectors(self, x, mod=True):
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
    def bindTwoVectors(self, x0,x1):
        xX = np.array(list(itertools.chain(x0,x1)),dtype='float32')
        return(xX) 
    
    def capitalRatio(self, tweet):
    
        firstCap, allCap = 0, 0
        length = len(tweet)

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
            print(m)
            models.append( m )
            
            #print(filename + ' loaded')
        return(models)
    
    
    def _load_unique_tokens(self, filename = 'uniqueTokens.dump'):
    
        filename = os.path.join(os.path.dirname(__file__),filename)
        return(joblib.load(filename))
        

    def _compare_tweets(self, X, classifiers):
        
        #feature_set = {emo: int(clf.predict(X)) for emo,clf in zipself._(self.emoNames, classifiers)} 
        feature_set = {emo: int(clf.predict(X[emo])) for emo,clf in zip(self.emoNames, classifiers)} 

        return feature_set        
    
    
    def analyse(self, **params):
        print("entering analyse")
        print(os.getcwd())
        print(os.path.dirname(__file__))
        print(__file__)
        logger.debug("Analysing with params {}".format(params))
        
        text_input = params.get("input", None)        
        DATA_FORMAT = 'weng'
        ESTIMATOR = 'LinearSVC'
        #LANG = params.get("language", None)         
        
        print("about to preprocess '%s'"%text_input)
        text = self._text_preprocessor(text_input)
        
        
        print("about to convert to vector")
        X = self._convert_text_to_vector(text, self._Dictionary, DATA_FORMAT)
        
        # load classifiers     
        print("about to load classifiers")
        classifiers = self._load_classifiers(DATA_FORMAT=DATA_FORMAT, ESTIMATOR=ESTIMATOR, emoNames=self.emoNames)
        
        feature_text = self._compare_tweets(X=X, classifiers=classifiers)
        
        print("classification done, about to build results object")
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
