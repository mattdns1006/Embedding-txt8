import tensorflow as tf
import pandas as pd
import os, gzip, pdb, string, pickle,pdb
import time
from datetime import datetime
import gensim.downloader as api

#Preprocessing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *
#from nltk.stem import WordNetLemmatizer as WNL # to try
from bs4 import BeautifulSoup

def sentence_clean(sentence,model_dir,first_n=None,stem=False):
    print("Cleaning")
    clean_start = time.time()
    sentence = BeautifulSoup(sentence,"html5lib").get_text()  
    letters_only = re.sub("[^a-zA-Z]", " ", sentence) #Remove non-letters
    words = letters_only.lower().split()    #Convert to lower case, split into individual words
    words = words[:] if first_n==None else words[:first_n]
    if stem==True:
        print("Stemming")
        stemmer = PorterStemmer()
        words_stemmed = list(map(stemmer.stem,words)) # expensive
        df_analyse = pd.DataFrame({'before':words,'after':words_stemmed}) # see how stemming works
        df_analyse.sample(200).to_csv(model_dir+"before_after_stem.csv")
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] #Remove stop words
    clean_end = time.time()
    time_taken = clean_end - clean_start
    print("Finished cleaning. Took {0:.2f} seconds. Number of words before/after stop word removal = {1}/{2}.".format(time_taken,len(words),len(meaningful_words)))
    return( " ".join( meaningful_words )) #Join the words back into one string separated by space

# ### Load data: text8 wikipedia dump (http://mattmahoney.net/dc/textdata.html)
def load_data(clean,model_dir,first_n,stem):
    txt8_clean_path = 'txt8_clean'
    if not os.path.exists(txt8_clean_path) or clean == True:
        print("Loading raw.")
        path = os.path.expanduser("~") + "/gensim-data/text8/text8.gz"
        if not os.path.exists: #download it
            api.download('txt8')
        with open(path) as f:
            f = gzip.open(path, 'rb')
            txt8_data = f.read()
            f.close()
        txt8_data_clean = [sentence_clean(txt8_data,model_dir,first_n,stem)] # Clean - takes a while
        with open(txt8_clean_path,"wb") as fp:
            pickle.dump(txt8_data_clean,fp)
    else:
        print("{0} found!".format(txt8_clean_path))
        with open(txt8_clean_path,"rb") as fp:
            txt8_data_clean = pickle.load(fp)

    return txt8_data_clean

def threads(clean_data):
    pdb.set_trace()
    

if __name__ == "__main__":
    text_clean = load_data()
    threads(text_clean)

