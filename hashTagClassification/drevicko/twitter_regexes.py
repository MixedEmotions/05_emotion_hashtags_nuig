
# coding: utf-8

# ## Twitter Regular Expressions
# Regular expressions for pre-processing pro-ana tweets. Includes:
# 
#  - surrounding LIWC-aware punctuation with spaces
#  - (except for url's and twitter entities)
#  - regexes for assorted smileys into 4 tokens: ): (: (; :/
#  - case munging: lower case unless all caps
#  
# The format is a list of `('search expression','replacement expression')` tuples.

# In[2]:

from __future__ import unicode_literals

try:
    re
except NameError:
    import regex as re
    print ("imported regex as re")

try:
    from ttp import ttp
except ImportError:
    ttp = None
    print("Couldn't load twitter-text! Tweet entities may not be recognised! Try `pip install twitter-text-python`")

try:
    # Python 2.6-2.7 
    from HTMLParser import HTMLParser
    unescape = HTMLParser().unescape
except ImportError:
    # Python 3.5
    from html import unescape

from itertools import chain # is this the same in python 2?
from operator import itemgetter


# In[2]:

punctuationRe = r""" (['`\[\](){}⟨⟩:,\-‒–—―!.?‘’“”"•;…/\|=+_~@#^&*<>]) """

regexStyles = ["twitterProAna","wordsAndPunct","walworth","twitterRuby"]
def setupRegexes(style="twitterProAna"):
  if style == regexStyles[0]: # twitterProAna or twitterEmoji
    def separateNumbers(m):
        if separateNumbers.string != m.string:
            separateNumbers.string = m.string
            separateNumbers.notParsing = False
        s=m.group()
        if s in set(('http','@','#')):
            separateNumbers.notParsing = True
            return s
        if s == ' ':
            separateNumbers.notParsing = False
            return s
        if separateNumbers.notParsing:
            return s
        return ' '+s+' '
    separateNumbers.string = None
    separateNumbers.searchString = r"http|@|#|\d+|\s"
    
    def skipEntities(m):
        
        return ' '+s+' '
    LIWC_punct = r"""'`\[\](){}⟨⟩:,\-‒–—―!.?‘’“”"•;…/\|=+_~@#^&*<>"""
    cleaningReList = [                       # surround punctuation with spaces (needs regex module, re module doesn't work!)
    (r"(?V1)(["+LIWC_punct+"||[^\p{Letter}\p{Separator}\p{Number}]])",r" \1 "),
                      # standardise quote characters
    (u"['`‘’“”\"]","'"),
                      # standardise smileys
    (r" [:=;] ( [\-Do'`‘’xPpLCc"'"'r"/,~] )?( [(\[{] )+"," ): "),
    (r" [:=] ( [\-Do'`‘’xPpLCc"'"'r"/,~] )?( [)\]}] )+"," (: "),
    (r" ; ( [\-Do'`‘’xPpLCc"'"'r"/~] )?( [)\]}] )+|( [(\[{] )+( [\-Do'`‘’xPpLCc"'"'r"/~] )? ; "," (; "),
    (r"(?<!http)(?<!https) [:=] ( [\-Do'`‘’xPpLCc"'"'r"/,~] )? / ( / )*"," :/ "),
    (r" - ( [._] )+ - "," ): "),
    (r"( [(\[{] )+( [\-Do'`‘’xPpLCc"'"'r"/,~] )? [:=;] "," (: "),
    (r"( [)\]}] )+( [\-Do'`‘’xPpLCc"'"'r"/,~] )? [:=] "," ): "),
                      # reform heart emoticons
    (r"< 3","<3"),
    #(r"http :  /  / (\S*)",r"http://\1"),
                      # reform url's (ie: remove inserted spaces)
    (r"http(s)? :  /  / ((\S*) (\.) |(\S*) (/) )(\S*)",r"http\1://\3\4\5\6\7"),
    (r"http(s)?://(\S*)( (\.) (\S*)| (/) (\S*))",r"http\1://\2\4\5\6\7"),
    (r"http(s)?://(\S*)( (\.) (\S*)| (/) (\S*))",r"http\1://\2\4\5\6\7"),
    #(r"(?<=[_\W0-9])['`‘’]|['`‘’](?=[_\W0-9])"," ' "),
    #(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])",r" "),
                      # keep "can't" etc as two tokens "can" and "'t"
                      # TODO: !! don't should become 'do' and 't
    (r"(\w) ' (\w)",r"\1 '\2"),
                      # keep 
                      # separate words and numbers (when not in a url)
    (r"([#@]) (\w+)",r"\1\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"), # up to 5 underscores in a name or tag - I guess others are rare!
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (separateNumbers.searchString,separateNumbers),
                      # reform "<3" (heart symbol in text)
    (r" < *3 "," <3 "),
#     (r"(\S*)((?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z]))",lambda x:x.group()+('' if re.findall('http',x.group()) else ' ')),
                      # munge word case (ie. convert to lower case if not all caps) except if a url
    (r"\S*([a-z]'?[A-Z]|[A-Z]'?[a-z])\w*",lambda x:x.group() if re.findall('http',x.group()) else x.group().lower())
#     (r"([^\p{Letter}\p{Separator}"+LIWC_punct+"])",r" \1 ")\
#     (r"([^\w\s"+LIWC_punct+"])",r" \1 ")\
    ]
  elif style == regexStyles[3]:
    def separateNumbers(m):
        if separateNumbers.string != m.string:
            separateNumbers.string = m.string
            separateNumbers.notParsing = False
        s=m.group()
        if s in set(('http','@','#')):
            separateNumbers.notParsing = True
            return s
        if s == ' ':
            separateNumbers.notParsing = False
            return s
        if separateNumbers.notParsing:
            return s
        return ' '+s+' '
    separateNumbers.string = None
    separateNumbers.searchString = r"http|@|#|\d+|\s"
    
    def skipEntities(m):
        
        return ' '+s+' '
    LIWC_punct = r"""'`\[\](){}⟨⟩:,\-‒–—―!.?‘’“”"•;…/\|=+_~@#^&*<>"""
    cleaningReList = [                       # surround punctuation with spaces (needs regex module, re module doesn't work!)
    (r"(?V1)(["+LIWC_punct+"||[^\p{Letter}\p{Separator}\p{Number}]])",r" \1 "),
                      # standardise quote characters
    (u"['`‘’“”\"]","'"),
                      # standardise smileys
    (r" [:=;] ( [\-Do'`‘’xPpLCc"'"'r"/,~] )?( [(\[{] )+"," ): "),
    (r" [:=] ( [\-Do'`‘’xPpLCc"'"'r"/,~] )?( [)\]}] )+"," (: "),
    (r" ; ( [\-Do'`‘’xPpLCc"'"'r"/~] )?( [)\]}] )+|( [(\[{] )+( [\-Do'`‘’xPpLCc"'"'r"/~] )? ; "," (; "),
    (r"(?<!http)(?<!https) [:=] ( [\-Do'`‘’xPpLCc"'"'r"/,~] )? / ( / )*"," :/ "),
    (r" - ( [._] )+ - "," ): "),
    (r"( [(\[{] )+( [\-Do'`‘’xPpLCc"'"'r"/,~] )? [:=;] "," (: "),
    (r"( [)\]}] )+( [\-Do'`‘’xPpLCc"'"'r"/,~] )? [:=] "," ): "),
                      # reform heart emoticons
    (r"< 3","<3"),
    #(r"http :  /  / (\S*)",r"http://\1"),
                      # reform url's (ie: remove inserted spaces)
    (r"http(s)? :  /  / ((\S*) (\.) |(\S*) (/) )(\S*)",r"http\1://\3\4\5\6\7"),
    (r"http(s)?://(\S*)( (\.) (\S*)| (/) (\S*))",r"http\1://\2\4\5\6\7"),
    (r"http(s)?://(\S*)( (\.) (\S*)| (/) (\S*))",r"http\1://\2\4\5\6\7"),
    #(r"(?<=[_\W0-9])['`‘’]|['`‘’](?=[_\W0-9])"," ' "),
    #(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])",r" "),
                      # keep "can't" etc as two tokens "can" and "'t"
                      # TODO: !! don't should become 'do' and 't
    (r"(\w) ' (\w)",r"\1 '\2"),
                      # keep 
                      # separate words and numbers (when not in a url)
    (r"([#@]) (\w+)",r"\1\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"), # up to 5 underscores in a name or tag - I guess others are rare!
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (r"([#@]\w+) _ (\w+)",r"\1_\2"),
    (separateNumbers.searchString,separateNumbers),
                      # reform "<3" (heart symbol in text)
    (r" < *3 "," <3 "),
#     (r"(\S*)((?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z]))",lambda x:x.group()+('' if re.findall('http',x.group()) else ' ')),
                      # munge word case (ie. convert to lower case if not all caps) except if a url
    (r"\S*([a-z]'?[A-Z]|[A-Z]'?[a-z])\w*",lambda x:x.group() if re.findall('http',x.group()) else x.group().lower())
#     (r"([^\p{Letter}\p{Separator}"+LIWC_punct+"])",r" \1 ")\
#     (r"([^\w\s"+LIWC_punct+"])",r" \1 ")\
    ]
  elif style == regexStyles[1] or style == regexStyles[2] : # "wordsAndPunct" or "walworth"
    cleaningReList = [      (r"""(['`\[\](){}⟨⟩:,\-‒–—―!.?‘’“”"•;…/\|=+_~@#^&*<>])""",r" \1 "),
      (u"['`‘’’“”\"]","'"),
      (u"[\-‒–—―]","-"),
      (r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])"," "),
      (r"(\w) ' (\w)",r"\1'\2")\
    ]
    if style == regexStyles[2]: # "walworth"
      cleaningReList.append((r"(-\s+)+",r"- "))
      cleaningReList.append((r".\s*$",r""))
  else:
    raise ValueError("Possible regexStyles: %s"%regexStyles) 
  return cleaningReList

regexList = setupRegexes()

def cleanString(regexList, text):
    '''Old cleanString function to support legacy code.'''
    text = unescape(text)
    if type(text) is str:
        try:
            text = text.decode('utf-8')
        except AttributeError:
            pass # python 3 strings don't do 'decode', but should be ok, so no need to do it anyway
    for regex in regexList:
        text = re.sub(regex[0],regex[1],text)
    return text


# In[ ]:

try:
    tweetParser = ttp.Parser(include_spans=True)
except AttributeError:
    tweetParser = None

# ttpParserLookup
    
def tweetPreprocessor(text, entitiesToDetect=("urls", "users", "lists", "tags")):
    """Takes a string, returns tuples containing either 
    (True, text_needing_parsing) or (False, entity_text_dont_parse)
    This relies on the ttp module for parsing tweets. If that module not present, it will silently pass the 
    whole text with "True".
    """
    try:
        entities = tweetParser.parse(text)
    except AttributeError:
        return (True, text)
    
    spans = []
    for label in entitiesToDetect:
        spanList = getattr(entities, label)
        if spanList:
            if label == 'lists':
                # lists are returned as a 3-tuple (name, user, (span)), we discard the user
                spanList = [(span[0], span[2]) for span in spanList] 
            spans.extend(spanList)
    idx = 0
    for span in sorted(spans, key=itemgetter(1)):
        entityStart, entityEnd = span[1]
        startString = text[idx:entityStart]
        if startString:
            yield (True, startString)
        ent = text[entityStart:entityEnd]
        if ent:
            yield (False, ent)
        idx = entityEnd
    endString = text[idx:]
    if endString:
        yield (True, endString)

def tokenize(text, regexList=regexList, preprocessor=tweetPreprocessor):
    """Tokenize a string, returning an iterator over tokens as strings.
    text : the string to be tokenized
    regexList : a list of (regex,replaceString) tuples, defaults to tweet specific processing
    preprocessor : a generator function preprocessor(text) which returns (boolean,substring) tuples, 
                 the boolean indicating if regexes should be applied. If None, apply regexes to original string.
    
    After applying regexes, the resulting string(s) are split on whitespace and yielded. Substrings returned by 
    the preprocessor with False are yielded as is (no regexes, no split)
    """
    subStringIter = preprocessor(text) if preprocessor else (True, text)
    for cleanIt, subString in subStringIter:
        if cleanIt:
            for word in cleanString(regexList, subString).split():
                yield word
        else:
            yield subString


# In[6]:

__doc__ = """
Defined functions setupRegexes(style=\"twitterProAna\"), cleanString(regexList, text) and tokenize(text).
Available regexStyles are:
"""
for s in regexStyles:
  __doc__ += "    %s\n"%s

