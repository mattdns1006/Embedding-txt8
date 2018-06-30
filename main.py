import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas as pd
import os, pdb, re, pdb, string, pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

#Preprocessing
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import keras.preprocessing.text as text
from keras.preprocessing import sequence

#Visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.3, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 5, "Batch size.")
flags.DEFINE_integer("embedding_size", 48, "Size of word embedding layer.")
flags.DEFINE_boolean("load", True, "Load previous checkpoint?")
flags.DEFINE_boolean("train", True, "Training model.")
flags.DEFINE_boolean("inference", True, "Inference.")
flags.DEFINE_integer("epochs", 50, "Number of training epochs.")

model_path = os.path.join(os.getcwd(),'model.ckpt')

# # Preprocessing 

def sentence_clean(sentence):
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
                if self.k == self.batch_size - 1:
                    self.k = 0
                    yield batch
                    batch = self.new_batch()
                    self.total_examples_seen += self.batch_size
                else:
                    self.k += 1

            self.epoch += 1

# ### Load data: text8 wikipedia dump (http://mattmahoney.net/dc/textdata.html)
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

#### Data object initiazation
data_obj = Data_obj(batch_size=FLAGS.batch_size,clean_data=txt8_data_clean)
generate_batch = data_obj.generator()

# # Model
#### Graph
vocabulary_size = data_obj.vocab_size
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_context = tf.placeholder(tf.int32, shape=[None, 1])
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, FLAGS.embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#NCE Loss
nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, FLAGS.embedding_size],
                            stddev=1.0 / np.sqrt(FLAGS.embedding_size)))
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
if FLAGS.train == True:
    print("Training")
    with tf.Session() as sess:
        if FLAGS.load == True:
            saver.restore(sess,model_path)
            print("Restored {0}.".format(model_path))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        embeddings_before = embeddings.eval()

        cur_losses = []

        while True:
            data = next(generate_batch)
            feed_dict = {train_inputs: data[:,0],train_context:data[:,[1]],learning_rate:FLAGS.learning_rate}
            _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
            cur_losses.append(cur_loss)
            if data_obj.total_examples_seen % 1000000 == 0 and data_obj.total_examples_seen > 0:
                print("{0} seen with running loss of {1:.3f}. Current epoch = {2}. Current LR = {3:.3f}".format(
                    data_obj.total_examples_seen,
                    np.mean(cur_losses),
                    data_obj.epoch,
                    FLAGS.learning_rate))
                cur_losses = []
                FLAGS.learning_rate /= 1.01
                save_path = saver.save(sess,model_path)
            if data_obj.epoch == FLAGS.epochs:
                print("Finished.")
                break


# ### Top words and their predicted counterparts
if FLAGS.inference == True:
    print("Inference")
    with tf.Session() as sess:
        saver.restore(sess,model_path)
        learnt_embeddings = embeddings.eval()
        top_n_words = 5 

        for word in ["education","port","america","three","philosophy","social","state"]:
            word_no = data_obj.Tokenizer.word_index[word]
            feed_dict={train_inputs:np.array([word_no])}
            word_embed, word_pred = sess.run([embed,soft_max],feed_dict)
            word_pred = word_pred.squeeze()
            top_n_args = word_pred.argsort()[-top_n_words:]
            print("Word = {0}".format(word))
            print(data_obj.inverse_tokenizer_sentence(top_n_args))
            print("\n")
        
    # # Visulization
    for perplexity in range(5,20,2):
        n_words_display = 80 # look at first n_words_display embedded
        tsne = TSNE(n_components=2,perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(learnt_embeddings[1:n_words_display+1]) #first embedding is meaningless (cant index it)
        labels = [data_obj.inverse_tokenizer(word_no) for word_no in range(1,vocabulary_size)[:n_words_display]]

        plt.figure(figsize=(25,25))
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

        plt.savefig("visualization_perplex_{0}.png".format(perplexity))
        plt.close()
