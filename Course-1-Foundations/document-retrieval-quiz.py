
# coding: utf-8

# # Document retrieval from wikipedia data

# ## Fire up GraphLab Create

# In[1]:

import graphlab


# # Load some text data - from wikipedia, pages on people

# In[2]:

people = graphlab.SFrame('people_wiki.gl/')


# Data contains:  link to wikipedia article, name of person, text of article.

# In[3]:

people.head()


# In[4]:

len(people)


# # Explore the dataset and checkout the text it contains
# 
# ## Exploring the entry for Elton John

# In[5]:

john = people[people['name'] == 'Elton John']


# In[6]:

john


# In[7]:

john['text']


# ## Exploring the entry for actor George Clooney

# In[8]:

#clooney = people[people['name'] == 'George Clooney']
#clooney['text']


# # Get the word counts for Elton John article

# In[9]:

john['word_count'] = graphlab.text_analytics.count_words(john['text'])


# In[10]:

print john['word_count']


# ## Sort the word counts for the Elton John article

# ### Turning dictonary of word counts into a table

# In[26]:

john_word_count_table = john[['word_count']].stack('word_count', new_column_name = ['word','count'])


# ### Sorting the word counts to show most common words at the top

# In[27]:

john_word_count_table.head()


# In[13]:

john_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# # Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[14]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[15]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf


# In[16]:

people['tfidf'] = tfidf


# ## Examine the TF-IDF for the Elton John article

# In[28]:

john = people[people['name'] == 'Elton John']


# In[29]:

john[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# Words with highest TF-IDF are much more informative.

# # Manually compute distances between a few people
# 
# Let's manually compare the distances between the articles for a few famous people.  

# In[35]:

paul = people[people['name'] == 'Paul McCartney’']


# In[32]:

beckham = people[people['name'] == 'Victoria Beckham']


# ## Is Elton closer to Paul McCartney than to Victoria Beckham?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about Elton John is closer to the one about Paul McCartney than that of Victoria Beckham.

# In[41]:

graphlab.distances.cosine(john['tfidf'][0],paul['tfidf'][0])


# In[40]:

graphlab.distances.cosine(john['tfidf'][0],beckham['tfidf'][0])


# In[45]:

#knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')

knn_word_count_model = graphlab.nearest_neighbors.create(people, features=['word_count'], label='name', distance='cosine')


# # Applying the nearest-neighbors model for retrieval

# ## Who is closest to Elton John?

# In[46]:

knn_word_count_model.query(john)


# In[47]:

knn_tfidf_model = graphlab.nearest_neighbors.create(people, features=['tfidf'], label='name', distance='cosine')


# In[48]:

knn_tfidf_model.query(john)


# ## Analysis for Victoria Beckham

# In[49]:

knn_word_count_model.query(beckham)


# In[51]:

knn_tfidf_model.query(beckham)


# In[ ]:



