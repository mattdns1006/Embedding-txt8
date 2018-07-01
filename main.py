import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas as pd
import os, pdb, re, pdb, string, pickle
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
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
flags.DEFINE_float("learning_rate", 0.8, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 5, "Batch size.")
flags.DEFINE_integer("embedding_size", 48, "Size of word embedding layer.")
flags.DEFINE_boolean("load", True, "Load previous checkpoint?")
flags.DEFINE_boolean("train", True, "Training model.")
flags.DEFINE_boolean("inference", True, "Inference.")
flags.DEFINE_integer("n_epochs", 50, "Number of training epochs.")
flags.DEFINE_string("model_path", "model.ckpt", "Model path.")


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



# # Model
class Model():
    def __init__(self,model_path,data_obj,embedding_size,lr,n_epochs):

        self.data_obj = data_obj
        self.model_path = os.path.join(os.getcwd(),model_path)
        self.vocabulary_size = data_obj.vocab_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.n_epochs = n_epochs

    def graph(self):
        #### Graph
        self.train_inputs = tf.placeholder(tf.int32, shape=[None])
        self.train_context = tf.placeholder(tf.int32, shape=[None, 1])
        self.embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

        #NCE Loss
        nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / np.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        hidden_out = tf.matmul(self.embed, tf.transpose(nce_weights)) + nce_biases
        self.soft_max = tf.nn.softmax(hidden_out)
        self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self.train_context,
                               inputs=self.embed,
                               num_sampled=1,
                               num_classes=self.vocabulary_size))
        # Optimization 
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def train(self,load):
        self.graph()
        with tf.Session() as sess:
            if load == True:
                self.saver.restore(sess,self.model_path)
                print("Restored {0}.".format(self.model_path))
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
            self.embeddings_before = self.embeddings.eval()
            cur_losses = []
            batch_generater = self.data_obj.generator()
            print("Training")
            while True:
                data = next(batch_generater)
                feed_dict = {self.train_inputs: data[:,0],self.train_context:data[:,[1]],self.learning_rate:self.lr}
                _, cur_loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                cur_losses.append(cur_loss)
                if self.data_obj.total_examples_seen % 100000 == 0 and self.data_obj.total_examples_seen > 0:
                    self.saver.save(sess,self.model_path)
                    print("{0} seen with running loss of {1:.3f}. Current epoch = {2}. Current LR = {3:.3f}. Saved in {4}.".format(
                        self.data_obj.total_examples_seen,
                        np.mean(cur_losses),
                        self.data_obj.epoch,
                        self.lr,
                        self.model_path))
                    cur_losses = []
                    self.lr/= 1.001

                if self.data_obj.epoch == self.n_epochs:
                    print("Finished.")
                    break

    def inference_examples(self):
        tf.reset_default_graph()
        self.graph()
        with tf.Session() as sess:
            self.saver.restore(sess,self.model_path)
            print("Restored {0}.".format(self.model_path))

            top_n_words = 10
            for word in ["education","port","america","three","philosophy","social","state"]:
                word_no = self.data_obj.Tokenizer.word_index[word]
                feed_dict={self.train_inputs:np.array([word_no])}
                word_embed, word_pred = sess.run([self.embed,self.soft_max],feed_dict)
                word_pred = word_pred.squeeze()
                top_n_args = word_pred.argsort()[-top_n_words:]
                print("Word = {0}".format(word))
                print(self.data_obj.inverse_tokenizer_sentence(top_n_args))
                print("\n")

    def visualize(self): 
        tf.reset_default_graph()
        self.graph()
        with tf.Session() as sess:
            self.saver.restore(sess,self.model_path)
            learnt_embeddings = self.embeddings.eval()

        folder = "vis/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for perplexity in range(5,20,2):
            n_words_display = 80 # look at first n_words_display embedded
            tsne = TSNE(n_components=2,perplexity=perplexity)
            reduced_embeddings = tsne.fit_transform(learnt_embeddings[1:n_words_display+1]) #first embedding is meaningless (cant index it)
            labels = [self.data_obj.inverse_tokenizer(word_no) for word_no in range(1,self.vocabulary_size)[:n_words_display]]

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

            save_path = folder + "visualization_perplex_{0}.png".format(perplexity)
            plt.savefig(save_path)
            print("Saved {0}.".format(save_path))
            plt.close()

#### Data object initiazation
if __name__ == "__main__":
    data_obj = Data_obj(batch_size=FLAGS.batch_size,clean_data=txt8_data_clean)
    model_obj = Model(
            model_path=FLAGS.model_path,
            data_obj=data_obj,
            embedding_size=FLAGS.embedding_size,
            lr=FLAGS.learning_rate,
            n_epochs=FLAGS.n_epochs)
    if FLAGS.train == True:
        model_obj.train(load=FLAGS.load)
    if FLAGS.inference== True:
        model_obj.inference_examples()
        model_obj.visualize()


# ### Top words and their predicted counterparts

            

