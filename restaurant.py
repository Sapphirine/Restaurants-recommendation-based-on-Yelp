
# coding: utf-8

# # Document Clustering and Topic Modeling

# In this project, we use unsupervised learning models to cluster unlabeled documents into different groups, visualize the results and identify their latent topics/structures.

# ## Contents

# <ul>
# <li>[Part 1: Load Data](#Part-1:-Load-Data)
# <li>[Part 2: Tokenizing and Stemming](#Part-2:-Tokenizing-and-Stemming)
# <li>[Part 3: TF-IDF](#Part-3:-TF-IDF)
# <li>[Part 4: K-means clustering](#Part-4:-K-means-clustering)
# </ul>

# # Part 1: Load Data

# In[2]:

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os
import csv

from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import lda


# Read data from files. In summary, we have 100 titles and 100 synoposes (combined from imdb and wiki).

# In[11]:
reviewfile =  open('review.csv')
busfile = open('business.csv')
readReview = csv.reader(reviewfile, delimiter=',') 
readBus = csv.reader(busfile, delimiter=',') 
ids = []
reviews = []  
count = 1  
for row in readReview:
    count = count + 1
    if count%100 == 0:
        ureview = row[2].decode('utf-8')
        review = ureview.encode('ascii','ignore')
        reviews.append(review)
for row in readBus:
    uid = row[23].decode('utf-8')
    id = uid.encode('ascii','ignore')
    ids.append(id)
    
#Because these synopses have already been ordered in popularity order, 
#we just need to generate a list of ordered numbers for future usage.
ids = ids[:200]
reviews = reviews[:200]

#  readCSV = csv.reader(csvfile, delimiter=',') 
#  ids = []
#  reviews = []    

#  for row in readCSV:
#    id = row[0]
#    ureview = row[2].decode('utf-8')
#    review = ureview.encode('ascii','ignore')
#    ids.append(id)
#    reviews.append(review)
    
#Because these synopses have already been ordered in popularity order, 
#we just need to generate a list of ordered numbers for future usage.
#ids = ids[:100]
#reviews = reviews[:100]
#print len(reviews)
#print reviews[1]
#print ids[1]
ranks = range(len(ids))


# # Part 2: Tokenizing and Stemming

# Load stopwords and stemmer function from NLTK library.
# Stop words are words like "a", "the", or "in" which don't convey significant meaning.
# Stemming is the process of breaking a word down into its root.


# Use nltk's English stopwords.
stopwords = nltk.corpus.stopwords.words('english')

print "We use " + str(len(stopwords)) + " stop-words from nltk library."
print stopwords[:50]


# In[13]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenization_and_stemming(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
#     tokens=[]
#     for sent in nltk.sent_tokenize(text):
#         for word in nltk.word_tokenize(sent):
#             if word not in stopwords:
#                 tokens.append(word);   
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenization(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[6]:

#tokenization_and_stemming("Mr Hoagie is an institution. Walking in, it does seem like a throwback to 30 years ago, old fashioned menu board, booths out of the 70s, and a large selection of food. Their speciality is the Italian Hoagie, and it is voted the best in the area year after year. I usually order the burger, while the patties are obviously cooked from frozen, all of the other ingredients are very fresh. Overall, its a good alternative to Subway, which is down the road.")


# Use our defined functions to analyze (i.e. tokenize, stem) our synoposes.

# In[14]:

docs_stemmed = []
docs_tokenized = []
for i in reviews:
    tokenized_and_stemmed_results = tokenization_and_stemming(i)
    docs_stemmed.extend(tokenized_and_stemmed_results)
    
    tokenized_results = tokenization(i)
    docs_tokenized.extend(tokenized_results)


# Create a mapping from stemmed words to original tokenized words for result interpretation.

# In[15]:

vocab_frame_dict = {docs_stemmed[x]:docs_tokenized[x] for x in range(len(docs_stemmed))}
#print vocab_frame_dict


# # Part 3: TF-IDF

# In[16]:

#define vectorizer parameters
tfidf_model = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenization_and_stemming, ngram_range=(1,1))

tfidf_matrix = tfidf_model.fit_transform(reviews) #fit the vectorizer to reviews

print "In total, there are " + str(tfidf_matrix.shape[0]) +       " reviews and " + str(tfidf_matrix.shape[1]) + " terms."


# In[17]:

tfidf_model.get_params()


# Save the terms identified by TF-IDF.

# In[18]:

tf_selected_words = tfidf_model.get_feature_names()
#print tf_selected_words


# # Part 4: K-means clustering

# In[19]:

from sklearn.cluster import KMeans

num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


# ## 4.1. Analyze K-means Result

# In[20]:

# create DataFrame films from all of the input files.
restaurants = { 'id': ids, 'rank': ranks, 'review': reviews, 'cluster': clusters}
frame = pd.DataFrame(restaurants, index = [clusters] , columns = ['rank', 'id', 'cluster'])


# In[21]:

frame.head(10)


# In[22]:

print "Number of films included in each cluster:"
frame['cluster'].value_counts().to_frame()


# In[ ]:




# In[23]:

print "<Document clustering result by K-means>"

#km.cluster_centers_ denotes the importances of each items in centroid.
#We need to sort it in decreasing-order and get the top k items.
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

Cluster_keywords_summary = {}
for i in range(num_clusters):
    print "Cluster " + str(i) + " words:" ,
    Cluster_keywords_summary[i] = []
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        Cluster_keywords_summary[i].append(vocab_frame_dict[tf_selected_words[ind]])
        print vocab_frame_dict[tf_selected_words[ind]] + ",",
    print
    #Here ix means index, which is the clusterID of each item.
    #Without tolist, the values result from dataframe is <type 'numpy.ndarray'>
    cluster_restaurants = frame.ix[i]['id'].values.tolist()
    print "Cluster " + str(i) + " ids (" + str(len(cluster_restaurants)) + " restaurants): " 
    print ", ".join(cluster_restaurants)
    print


# In[ ]:




# ## 4.2. Plot K-means Result

# In[24]:

pca = decomposition.PCA(n_components=2)
tfidf_matrix_np=tfidf_matrix.toarray()
pca.fit(tfidf_matrix_np)
X = pca.transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:, 1]

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5:'#cd7f32',6:'#545454',7:'#3299cc',8:'#000000'}
#set up cluster names using a dict
cluster_names = {}
for i in range(num_clusters):
    cluster_names[i] = ", ".join(Cluster_keywords_summary[i])


# In[25]:

#get_ipython().magic(u'matplotlib inline')

#create data frame with PCA cluster results
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, id=ids)) 
groups = df.groupby(clusters)

# set up plot
fig, ax = plt.subplots(figsize=(16, 9))
#Set color for each cluster/group
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')

ax.legend(numpoints=1,loc=4)  #show legend with only 1 point, position is right bottom.

plt.show() #show the plot

plt.close()

