import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas as pd
import os, pdb, re, pdb
import string
import matplotlib
import matplotlib.pyplot as plt
import pickle

#Preprocessing
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import keras.preprocessing.text as text
from keras.preprocessing import sequence

#Visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#Display options
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(linewidth=200,threshold=np.nan,formatter={'float_kind':float_formatter})
pd.set_option("display.max_colwidth",200)
pd.set_option("display.max_rows",200)

N_OUT = 5
BATCH_SIZE = 5
HIDDEN_SIZE = 48 
NUM_LAYERS = 2
INIT_SCALE = 0.05
lr = 0.3
LOAD = True
TRAIN = True
INFERENCE = True
model_path = os.path.join(os.getcwd(),'model.ckpt')

# # Preprocessing 

# ### Load data: text8 wikipedia dump (http://mattmahoney.net/dc/textdata.html)

def sentence_clean( sentence ):
    review_text = BeautifulSoup(sentence,"html5lib").get_text()  
    letters_only = re.sub("[^a-zA-Z]", " ", sentence) #Remove non-letters
    words = letters_only.lower().split()    #Convert to lower case, split into individual words                          
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] #Remove stop words
    return( " ".join( meaningful_words )) #Join the words back into one string separated by space

class Data_obj():
    def __init__(self,batch_size,clean_data):
        self.epoch = 1
        self.batch_size = batch_size
        self.clean_data = clean_data
        
        self.Tokenizer = text.Tokenizer()
        self.Tokenizer.fit_on_texts(clean_data)
        self.words = self.Tokenizer.word_index.keys()
        self.encoded_text = self.Tokenizer.texts_to_sequences(self.clean_data)[0]
        
        self.inverse_tokenizer = lambda num: list(self.Tokenizer.word_index.keys())[list(self.Tokenizer.word_index.values()).index(num)] #inverse
        self.inverse_tokenizer_sentence = lambda sentence: list(map(self.inverse_tokenizer,sentence))
        
        self.vocab_size = len(self.Tokenizer.word_index) + 1
        print("There are {0} unique words in data set.".format(self.vocab_size))
           
    def new_batch(self):
        return np.zeros((self.batch_size,2)).astype(np.int32)
    
    def generator(self):
        self.i = self.k = 0
        batch = self.new_batch()
        self.total_examples_seen = 0
        while True:
            n_words = len(self.encoded_text)
            for j in range(n_words):
                context = self.encoded_text[j]
                if j == 0:
                    target = self.encoded_text[j+1]
                elif j == n_words - 1:
                    target = self.encoded_text[j-1]
                elif rng.uniform() < 0.5:
                    target = self.encoded_text[j-1]
                else:
                    target = self.encoded_text[j+1]
                    
                batch[self.k,0] = context
                batch[self.k,1] = target
                if self.k == BATCH_SIZE - 1:
                    self.k = 0
                    yield batch
                    batch = self.new_batch()
                    self.total_examples_seen += self.batch_size
                else:
                    self.k += 1

            self.epoch += 1


# ### Data object initiazation
txt8_clean_path = "txt8_clean" #path to cleaned data
if not os.path.exists(txt8_clean_path):
    print("{0} not found. Loading raw and cleaning.".format(txt8_clean_path))

    with open('/Users/matt/gensim-data/text8/text8') as f:
        txt8_data = f.read()
        print("Length of dataset in words = {0}.".format(len(txt8_data)))
        f.close()
    txt8_data_clean = [sentence_clean(txt8_data)] # Clean - takes a while

    with open(txt8_clean_path,"wb") as fp:
        pickle.dump(txt8_data_clean,fp)
else:
    print("{0} found!".format(txt8_clean_path))
    with open(txt8_clean_path,"rb") as fp:
        txt8_data_clean = pickle.load(fp)
print("Length of (cleaned) dataset in words = {0}.".format(len(txt8_data_clean[0])))

data_obj = Data_obj(batch_size=BATCH_SIZE,clean_data=txt8_data_clean)
generate_batch = data_obj.generator()

# # Model
#### Graph
vocabulary_size = data_obj.vocab_size
embedding_size = HIDDEN_SIZE
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_context = tf.placeholder(tf.int32, shape=[None, 1])
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#NCE Loss
nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
hidden_out = tf.matmul(embed, tf.transpose(nce_weights)) + nce_biases
soft_max = tf.nn.softmax(hidden_out)
loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=1,
                       num_classes=vocabulary_size))
# Optimization 
learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()


#cosine similarity
embedding_norm=tf.nn.l2_normalize(embeddings,axis=1)
similarity = tf.matmul(embedding_norm, tf.transpose(embedding_norm))


# ### Training
if TRAIN == True:
    with tf.Session() as sess:
        if LOAD == True:
            saver.restore(sess,model_path)
            print("Restored {0}.".format(model_path))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        embeddings_before = embeddings.eval()

        cur_losses = []

        while True:
            data = next(generate_batch)
            feed_dict = {train_inputs: data[:,0],train_context:data[:,[1]],learning_rate:lr}
            _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
            cur_losses.append(cur_loss)
            if data_obj.total_examples_seen % 100000 == 0 and data_obj.total_examples_seen > 0:
                print("{0} seen with running loss of {1:.3f}. Current epoch = {2}. Current lr = {3:.3f}".format(data_obj.total_examples_seen,np.mean(cur_losses),data_obj.epoch,lr))
                cur_losses = []
                lr /= 1.003
                save_path = saver.save(sess,model_path)
            if data_obj.epoch == 50:
                print("Finished.")
                break
        



# ### Top words and their predicted counterparts
def inference():
    with tf.Session() as sess:
        saver.restore(sess,model_path)
        top_n_words = 3
        learnt_embeddings = embeddings.eval()
        for word_no in range(1,vocabulary_size)[:10]:
            word = data_obj.inverse_tokenizer(word_no)
            feed_dict={train_inputs:np.array([word_no])}
            word_embed, word_pred = sess.run([embed,soft_max],feed_dict)
            word_pred = word_pred.squeeze()
            top_n_args = word_pred.argsort()[-top_n_words:]
            print("\n")
            print(word,word_no)
            #print(word_pred)
            print(data_obj.inverse_tokenizer_sentence(top_n_args))


    # # Visulization
    n_words_display = 60 # look at first n_words_display embedded
    pca = TSNE(n_components=2,perplexity=5)
    reduced_embeddings = pca.fit_transform(learnt_embeddings[1:n_words_display+1]) #first embedding is meaningless (cant index it)
    labels = [data_obj.inverse_tokenizer(word_no) for word_no in range(1,vocabulary_size)[:n_words_display]]

    plt.figure(figsize=(15,15))
    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker='o',
        cmap=plt.get_cmap('Spectral'))

    for label, x, y in zip(labels, reduced_embeddings[:, 0], reduced_embeddings[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    plt.savefig("visualization.png")
    plt.show()

if INFERENCE == True:
    inference()
