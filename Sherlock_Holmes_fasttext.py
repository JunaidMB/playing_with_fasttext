# Install packages
!conda install gensim 
!conda install nltk 
!conda install spacy 
!git clone https://github.com/facebookresearch/fastText.git
!pip install fasttext

# Import packages
import fasttext
import string
import re
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy
import gensim
import gensim
from gensim.utils import simple_preprocess
from collections import Counter
from datetime import datetime, timedelta
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim import models
from numpy import asarray
import unicodedata

# Load Files
## Scandal in Bohemia sentences
scandal_in_bohemia_sentences = open("scandal_in_bohemia_sentences.txt", "r")
scandal_in_bohemia_sentences = scandal_in_bohemia_sentences.readlines()

scandal_in_bohemia_sentences_no_stopwords = open("scandal_in_bohemia_sentences_no_stopwords.txt", "r")
scandal_in_bohemia_sentences_no_stopwords = scandal_in_bohemia_sentences_no_stopwords.readlines()


# Helper functions
## NLP Functions
def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
    return str(text)

def preprocess(text, remove_accents=False, lower = True, remove_less_than=0, remove_more_than=100, remove_punct=True, 
               remove_alpha=False, remove_stopwords=True, add_custom_stopwords = [], lemma=False, stem=False, remove_url=True):
    '''Tokenises and preprocesses text.
    Parameters: 
    text (string): a string of text
    remove_accents (boolean): removes accents
    lower (boolean): lowercases text
    remove_less_than (int): removes words less than X letters
    remove_more_than (int): removes words more than X letters
    remove_punct (boolean): removes punctuation 
    remove_alpha (boolean): removes non-alphabetic tokens
    remove_stopwords (boolean): removes stopwords
    add_custom_stopwords (list): adds custom stopwords
    lemma (boolean): lemmantises tokens
    stem (boolean): stems tokes using the Porter Stemmer
    Output: 
    tokens (list): a list of cleaning tokens
    '''
    if remove_accents == True:
        text = strip_accents(text)
    if lower == True:
        text = text.lower()
    if remove_url == True:
            text = re.sub(r'http\S+', '', text)
    #tokens = simple_preprocess(text, deacc=remove_accents, min_len=remove_less_than, max_len=remove_more_than)
    tokens = text.split()
    if remove_punct == True:
        tokens = [ch.translate(str.maketrans('', '', string.punctuation)) for ch in tokens]
    if remove_alpha == True:
        tokens = [token for token in tokens if token.isalpha()]
    if remove_stopwords == True:
        for i in add_custom_stopwords:
            stop_words.add(i)
        tokens = [token for token in tokens if not token in stop_words]
    tokens = [i for i in tokens if remove_less_than <=  len(i) <= remove_more_than]
    if lemma == True: 
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    if stem == True:
        tokens = [PorterStemmer().stem(token) for token in tokens]
    return tokens

# Build a word2vec model
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

# For word2vec, we need a list tokens
scandal_in_bohemia_tokens = [preprocess(i) for i in scandal_in_bohemia_sentences]
w2v_model = gensim.models.Word2Vec(scandal_in_bohemia_tokens, size = 500, window = 10, min_count=1, workers = 4)

# Build a fasttext model - one with and without subwords
# Fasttext takes as an input, a txt file in the environment - so we refer to the raw file and not the one in our Python environment
# The input file here is the same as the list of sentences we have already seen but with stop words removed

ft_model = fasttext.train_unsupervised('scandal_in_bohemia_sentences_no_stopwords.txt', minn = 2, maxn = 5, dim = 500)
ft_model_wo_subwords = fasttext.train_unsupervised('scandal_in_bohemia_sentences_no_stopwords.txt', maxn = 0, dim = 500)


# Comparing the outputs from each model

w2v_model.wv.most_similar('woman', topn = 20)
ft_model.get_nearest_neighbors('woman', k = 20)
ft_model_wo_subwords.get_nearest_neighbors('woman', k = 20)

