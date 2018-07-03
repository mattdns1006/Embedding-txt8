import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas as pd
import os, re, pdb, string, pickle
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('ggplot')
matplotlib.use("Agg")

#Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
#from nltk.stem import WordNetLemmatizer as WNL # to try
from bs4 import BeautifulSoup
import keras.preprocessing.text as text
import time
from keras.preprocessing import sequence

#Visualization
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("embedding_size", 48, "Size of word embedding layer.")
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 5, "Batch size.")
flags.DEFINE_integer("n_epochs", 50, "Number of training epochs.")
flags.DEFINE_boolean("clean", False, "Clean raw - eg if trying new preprocessing.")
flags.DEFINE_boolean("load", True, "Load previous checkpoint?")
flags.DEFINE_boolean("train", True, "Training model.")
flags.DEFINE_boolean("inference", True, "Inference.")
flags.DEFINE_boolean("visualize", True, "Visualize embeddings.")

# # Preprocessing 
def sentence_clean(sentence,stem=False):
    print("Cleaning")
    clean_start = time.time()
    review_text = BeautifulSoup(sentence,"html5lib").get_text()  
    letters_only = re.sub("[^a-zA-Z]", " ", sentence) #Remove non-letters
    words = letters_only.lower().split()    #Convert to lower case, split into individual words
    words = words[590000:600000]
    if stem==True:
        stemmer = PorterStemmer()
        words_stemmed = list(map(stemmer.stem,words)) # expensive
        df_analyse = pd.DataFrame({'before':words,'after':words_stemmed}) # see how stemming works
        df_analyse.sample(200).to_csv("before_after_stem.csv")
        words = words_stemmed[:]

    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] #Remove stop words
    clean_end = time.time()
    time_taken = clean_end - clean_start
    print("Finished cleaning. Took {0:.2f} seconds.".format(time_taken))
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
def load_data(clean=False):
    txt8_clean_path = "txt8_clean" #path to cleaned data
    if not os.path.exists(txt8_clean_path) or FLAGS.clean == True:
        print("Loading raw.")
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
    return txt8_data_clean

def makedirs(model_dir):
    dirs = ['models','vis']
    dirs.append(model_dir)
    [os.makedirs(dir) for dir in dirs if not os.path.exists(dir)]
        
# # Model
class Model():
    def __init__(self,model_dir,data_obj,embedding_size,lr,n_epochs):

        self.data_obj = data_obj
        self.model_dir = model_dir
        self.model_path = self.model_dir + "model.ckpt" 
        self.vocabulary_size = data_obj.vocab_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.labels_path = os.path.join(self.model_dir,'labels.tsv')
        labels = pd.DataFrame(list(self.data_obj.words))
        labels.to_csv(self.labels_path,sep='\t',index=False,header=True)

    def graph(self):
        #### Graph
        self.train_inputs = tf.placeholder(tf.int32, shape=[None])
        self.train_context = tf.placeholder(tf.int32, shape=[None, 1])

        with tf.name_scope("Embedding"):
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0),name="embedding_matrix")
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs,name="look_up")
        self.projector_config = projector.ProjectorConfig()
        embedding = self.projector_config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        #embedding.metadata_path = self.labels_path 
        embedding.metadata_path = "labels.tsv"


        #NCE Loss
        with tf.name_scope("Weights"):
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
        self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)
        self.saver = tf.train.Saver()

        with tf.name_scope("Summaries"):
            tf.summary.scalar('nce_loss',self.loss)
            tf.summary.scalar('learning_rate',self.learning_rate)
            tf.summary.histogram('histogram loss',self.loss)
        self.summary_op = tf.summary.merge_all()

    def train(self,load):

        self.graph()
        with tf.Session() as sess:
            # Summaries
            now = datetime.now()
            #writer_path = 'summary/{0}/'.format(now.strftime("%Y%m%d-%H%M%S"))
            writer  =  tf.summary.FileWriter(self.model_dir, tf.get_default_graph())
            projector.visualize_embeddings(writer, self.projector_config)
            if load == True:
                self.saver.restore(sess,self.model_path)
                print("Restored {0}.".format(self.model_path))
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
            cur_losses = []
            batch_generater = self.data_obj.generator()
            print("Training")
            step = 0
            while True:
                data = next(batch_generater)
                feed_dict = {self.train_inputs: data[:,0],self.train_context:data[:,[1]],self.learning_rate:self.lr}
                _, cur_loss,summary = sess.run([self.optimizer, self.loss, self.summary_op], feed_dict=feed_dict)
                cur_losses.append(cur_loss)
                writer.add_summary(summary, step)
                step += 1

                if self.data_obj.total_examples_seen % 10000 == 0 and self.data_obj.total_examples_seen > 0:
                    print("{0} seen with running loss of {1:.3f}. Current epoch = {2}. Current LR = {3:.3f}. ".format(
                        self.data_obj.total_examples_seen,
                        np.mean(cur_losses),
                        self.data_obj.epoch,
                        self.lr))
                    cur_losses = []
                    self.lr/= 1.001

                if self.data_obj.total_examples_seen % 2000000 == 0:
                    self.saver.save(sess,self.model_path)
                    print("Saved in {0}.".format(self.model_path))

                if self.data_obj.epoch == self.n_epochs:
                    print("Finished.")
                    break

    def inference(self,top_n_words=5,examples=True):
        tf.reset_default_graph()
        self.graph()
        with tf.Session() as sess:
            self.saver.restore(sess,self.model_path)
            print("Restored {0}.".format(self.model_path))
            def return_closest_words(word):
                word_no = self.data_obj.Tokenizer.word_index[word]
                feed_dict={self.train_inputs:np.array([word_no])}
                word_embed, word_pred = sess.run([self.embed,self.soft_max],feed_dict)
                word_pred = word_pred.squeeze()
                top_n_args = word_pred.argsort()[-top_n_words:]
                print("Word = {0}".format(word))
                print(self.data_obj.inverse_tokenizer_sentence(top_n_args))
                print("\n")
            if examples == True:
                [return_closest_words(word) for word in ["education","port","america","three","philosophy","social","state"]]
            while True:
                word = input("Enter word == >")
                try:
                    return_closest_words(word)
                except KeyError:
                    print("Not a valid word. Try again.")

    def visualize(self): 
        tf.reset_default_graph()
        self.graph()
        with tf.Session() as sess:
            self.saver.restore(sess,self.model_path)
            learnt_embeddings = self.embeddings.eval()

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
    model_dir = "models/model_{0}/".format(FLAGS.embedding_size)
    txt8_data_clean = load_data()
    makedirs(model_dir)
    data_obj = Data_obj(batch_size=FLAGS.batch_size,clean_data=txt8_data_clean)
    model_obj = Model(
            model_dir=model_dir,
            data_obj=data_obj,
            embedding_size=FLAGS.embedding_size,
            lr=FLAGS.learning_rate,
            n_epochs=FLAGS.n_epochs)
    if FLAGS.train == True:
        model_obj.train(load=FLAGS.load)
    if FLAGS.inference== True:
        model_obj.inference()
    if FLAGS.visualize== True:
        model_obj.visualize()
