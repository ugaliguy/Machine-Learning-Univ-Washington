
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
# ## Exploring the entry for president Obama

# In[5]:

obama = people[people['name'] == 'Barack Obama']


# In[6]:

obama


# In[7]:

obama['text']


# ## Exploring the entry for actor George Clooney

# In[8]:

clooney = people[people['name'] == 'George Clooney']
clooney['text']


# # Get the word counts for Obama article

# In[9]:

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])


# In[10]:

print obama['word_count']


# ## Sort the word counts for the Obama article

# ### Turning dictonary of word counts into a table

# In[11]:

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])


# ### Sorting the word counts to show most common words at the top

# In[12]:

obama_word_count_table.head()


# In[13]:

obama_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# # Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[14]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[19]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf


# In[20]:

people['tfidf'] = tfidf


# ## Examine the TF-IDF for the Obama article

# In[21]:

obama = people[people['name'] == 'Barack Obama']


# In[22]:

obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# Words with highest TF-IDF are much more informative.

# # Manually compute distances between a few people
# 
# Let's manually compare the distances between the articles for a few famous people.  

# In[23]:

clinton = people[people['name'] == 'Bill Clinton']


# In[24]:

beckham = people[people['name'] == 'David Beckham']


# ## Is Obama closer to Clinton than to Beckham?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about president Obama is closer to the one about former president Clinton than that of footballer David Beckham.

# In[25]:

graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])


# In[26]:

graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# # Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  

# In[27]:

knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# # Applying the nearest-neighbors model for retrieval

# ## Who is closest to Obama?

# In[28]:

knn_model.query(obama)


# As we can see, president Obama's article is closest to the one about his vice-president Biden, and those of other politicians.  

# ## Other examples of document retrieval

# In[31]:

swift = people[people['name'] == 'Taylor Swift']


# In[32]:

knn_model.query(swift)


# In[34]:

jolie = people[people['name'] == 'Angelina Jolie']


# In[35]:

knn_model.query(jolie)


# In[36]:

arnold = people[people['name'] == 'Arnold Schwarzenegger']


# In[37]:

knn_model.query(arnold)


# In[ ]:



