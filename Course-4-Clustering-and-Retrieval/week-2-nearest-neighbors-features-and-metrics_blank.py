
# coding: utf-8

# # Nearest Neighbors

# When exploring a large set of documents -- such as Wikipedia, news articles, StackOverflow, etc. -- it can be useful to get a list of related material. To find relevant documents you typically
# * Decide on a notion of similarity
# * Find the documents that are most similar 
# 
# In the assignment you will
# * Gain intuition for different notions of similarity and practice finding similar documents. 
# * Explore the tradeoffs with representing documents using raw word counts and TF-IDF
# * Explore the behavior of different distance metrics by looking at the Wikipedia pages most similar to President Obama’s page.

# **Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

# ## Import necessary packages

# As usual we need to first import the Python packages that we will need.
# 
# The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).

# In[1]:

import os
os.environ["OMP_NUM_THREADS"] = "1"
import graphlab

graphlab.SArray(range(1000)).apply(lambda x: x)


# In[2]:

#import graphlab
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic(u'matplotlib inline')

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'


# ## Load Wikipedia dataset

# We will be using the same dataset of Wikipedia pages that we used in the Machine Learning Foundations course (Course 1). Each element of the dataset consists of a link to the wikipedia article, the name of the person, and the text of the article (in lowercase).  

# In[3]:

wiki = graphlab.SFrame('people_wiki.gl')


# In[4]:

wiki


# ## Extract word count vectors

# As we have seen in Course 1, we can extract word count vectors using a GraphLab utility function.  We add this as a column in `wiki`.

# In[5]:

wiki['word_count'] = graphlab.text_analytics.count_words(wiki['text'])


# In[6]:

wiki


# ## Find nearest neighbors

# Let's start by finding the nearest neighbors of the Barack Obama page using the word count vectors to represent the articles and Euclidean distance to measure distance.  For this, again will we use a GraphLab Create implementation of nearest neighbor search.

# In[7]:

model = graphlab.nearest_neighbors.create(wiki, label='name', features=['word_count'],
                                          method='brute_force', distance='euclidean')


# Let's look at the top 10 nearest neighbors by performing the following query:

# In[8]:

model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)


# All of the 10 people are politicians, but about half of them have rather tenuous connections with Obama, other than the fact that they are politicians.
# 
# * Francisco Barrio is a Mexican politician, and a former governor of Chihuahua.
# * Walter Mondale and Don Bonker are Democrats who made their career in late 1970s.
# * Wynn Normington Hugh-Jones is a former British diplomat and Liberal Party official.
# * Andy Anstett is a former politician in Manitoba, Canada.
# 
# Nearest neighbors with raw word counts got some things right, showing all politicians in the query result, but missed finer and important details.
# 
# For instance, let's find out why Francisco Barrio was considered a close neighbor of Obama.  To do this, let's look at the most frequently used words in each of Barack Obama and Francisco Barrio's pages:

# In[9]:

def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)


# In[10]:

obama_words = top_words('Barack Obama')
obama_words


# In[11]:

barrio_words = top_words('Francisco Barrio')
barrio_words


# Let's extract the list of most frequent words that appear in both Obama's and Barrio's documents. We've so far sorted all words from Obama and Barrio's articles by their word frequencies. We will now use a dataframe operation known as **join**. The **join** operation is very useful when it comes to playing around with data: it lets you combine the content of two tables using a shared column (in this case, the word column). See [the documentation](https://dato.com/products/create/docs/generated/graphlab.SFrame.join.html) for more details.
# 
# For instance, running
# ```
# obama_words.join(barrio_words, on='word')
# ```
# will extract the rows from both tables that correspond to the common words.

# In[12]:

combined_words = obama_words.join(barrio_words, on='word')
combined_words


# Since both tables contained the column named `count`, SFrame automatically renamed one of them to prevent confusion. Let's rename the columns to tell which one is for which. By inspection, we see that the first column (`count`) is for Obama and the second (`count.1`) for Barrio.

# In[13]:

combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words


# **Note**. The **join** operation does not enforce any particular ordering on the shared column. So to obtain, say, the five common words that appear most often in Obama's article, sort the combined table by the Obama column. Don't forget `ascending=False` to display largest counts first.

# In[14]:

combined_words.sort('Obama', ascending=False)


# **Quiz Question**. Among the words that appear in both Barack Obama and Francisco Barrio, take the 5 that appear most frequently in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?
# 
# Hint:
# * Refer to the previous paragraph for finding the words that appear in both articles. Sort the common words by their frequencies in Obama's article and take the largest five.
# * Each word count vector is a Python dictionary. For each word count vector in SFrame, you'd have to check if the set of the 5 common words is a subset of the keys of the word count vector. Complete the function `has_top_words` to accomplish the task.
#   - Convert the list of top 5 words into set using the syntax
# ```
# set(common_words)
# ```
#     where `common_words` is a Python list. See [this link](https://docs.python.org/2/library/stdtypes.html#set) if you're curious about Python sets.
#   - Extract the list of keys of the word count dictionary by calling the [`keys()` method](https://docs.python.org/2/library/stdtypes.html#dict.keys).
#   - Convert the list of keys into a set as well.
#   - Use [`issubset()` method](https://docs.python.org/2/library/stdtypes.html#set) to check if all 5 words are among the keys.
# * Now apply the `has_top_words` function on every row of the SFrame.
# * Compute the sum of the result column to obtain the number of articles containing all the 5 top words.

# In[15]:

common_words = combined_words['word'][0:5]  # YOUR CODE HERE

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())   # YOUR CODE HERE
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return set(common_words).issubset(unique_words)  # YOUR CODE HERE

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
wiki['has_top_words'].sum() # YOUR CODE HERE


# **Checkpoint**. Check your `has_top_words` function on two random articles:

# In[18]:

print 'Output from your function:', has_top_words(wiki[32]['word_count'])
print 'Correct output: True'
print 'Also check the length of unique_words. It should be 167'


# In[19]:

print 'Output from your function:', has_top_words(wiki[33]['word_count'])
print 'Correct output: False'
print 'Also check the length of unique_words. It should be 188'


# **Quiz Question**. Measure the pairwise distance between the Wikipedia pages of Barack Obama, George W. Bush, and Joe Biden. Which of the three pairs has the smallest distance?
# 
# Hint: To compute the Euclidean distance between two dictionaries, use `graphlab.toolkits.distances.euclidean`. Refer to [this link](https://dato.com/products/create/docs/generated/graphlab.toolkits.distances.euclidean.html) for usage.

# In[20]:

def get_word_count_dict(name):
    return wiki[wiki['name'] == name]['word_count'][0]


# In[23]:

graphlab.toolkits.distances.euclidean(get_word_count_dict('Barack Obama'), get_word_count_dict('George W. Bush'))


# In[24]:

graphlab.toolkits.distances.euclidean(get_word_count_dict('Barack Obama'), get_word_count_dict('Joe Biden'))


# In[25]:

graphlab.toolkits.distances.euclidean(get_word_count_dict('George W. Bush'), get_word_count_dict('Joe Biden'))


# **Quiz Question**. Collect all words that appear both in Barack Obama and George W. Bush pages.  Out of those words, find the 10 words that show up most often in Obama's page. 

# In[26]:

bush_words = top_words('George W. Bush')


# In[27]:

obama_words.join(bush_words, on='word').rename({'count':'Obama', 'count.1':'Bush'}).sort('Obama', ascending=False)


# **Note.** Even though common words are swamping out important subtle differences, commonalities in rarer political words still matter on the margin. This is why politicians are being listed in the query result instead of musicians, for example. In the next subsection, we will introduce a different metric that will place greater emphasis on those rarer words.

# ## TF-IDF to the rescue

# Much of the perceived commonalities between Obama and Barrio were due to occurrences of extremely frequent words, such as "the", "and", and "his". So nearest neighbors is recommending plausible results sometimes for the wrong reasons. 
# 
# To retrieve articles that are more relevant, we should focus more on rare words that don't happen in every article. **TF-IDF** (term frequency–inverse document frequency) is a feature representation that penalizes words that are too common.  Let's use GraphLab Create's implementation of TF-IDF and repeat the search for the 10 nearest neighbors of Barack Obama:

# In[28]:

wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count'])


# In[29]:

model_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                 method='brute_force', distance='euclidean')


# In[30]:

model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)


# Let's determine whether this list makes sense.
# * With a notable exception of Roland Grossenbacher, the other 8 are all American politicians who are contemporaries of Barack Obama.
# * Phil Schiliro, Jesse Lee, Samantha Power, and Eric Stern worked for Obama.
# 
# Clearly, the results are more plausible with the use of TF-IDF. Let's take a look at the word vector for Obama and Schilirio's pages. Notice that TF-IDF representation assigns a weight to each word. This weight captures relative importance of that word in the document. Let us sort the words in Obama's article by their TF-IDF weights; we do the same for Schiliro's article as well.

# In[31]:

def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)


# In[32]:

obama_tf_idf = top_words_tf_idf('Barack Obama')
obama_tf_idf


# In[33]:

schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
schiliro_tf_idf


# Using the **join** operation we learned earlier, try your hands at computing the common words shared by Obama's and Schiliro's articles. Sort the common words by their TF-IDF weights in Obama's document.

# In[34]:

obama_tf_idf.join(schiliro_tf_idf, 'word')


# The first 10 words should say: Obama, law, democratic, Senate, presidential, president, policy, states, office, 2011.

# **Quiz Question**. Among the words that appear in both Barack Obama and Phil Schiliro, take the 5 that have largest weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?

# In[35]:

common_words = set(obama_tf_idf.join(schiliro_tf_idf, 'word').sort('weight', ascending=False)['word'][:5])

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(dict(word_count_vector).keys())
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words.issubset(unique_words)

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
wiki['has_top_words'].sum()


# Notice the huge difference in this calculation using TF-IDF scores instead  of raw word counts. We've eliminated noise arising from extremely common words.

# ## Choosing metrics

# You may wonder why Joe Biden, Obama's running mate in two presidential elections, is missing from the query results of `model_tf_idf`. Let's find out why. First, compute the distance between TF-IDF features of Obama and Biden.

# **Quiz Question**. Compute the Euclidean distance between TF-IDF features of Obama and Biden. Hint: When using Boolean filter in SFrame/SArray, take the index 0 to access the first match.

# In[36]:

def get_tfidf_dict(name):
    return wiki[wiki['name'] == name]['tf_idf'][0]


# In[37]:

graphlab.toolkits.distances.euclidean(get_tfidf_dict('Barack Obama'), get_tfidf_dict('Joe Biden'))


# The distance is larger than the distances we found for the 10 nearest neighbors, which we repeat here for readability:

# In[38]:

model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)


# But one may wonder, is Biden's article that different from Obama's, more so than, say, Schiliro's? It turns out that, when we compute nearest neighbors using the Euclidean distances, we unwittingly favor short articles over long ones. Let us compute the length of each Wikipedia document, and examine the document lengths for the 100 nearest neighbors to Obama's page.

# In[39]:

def compute_length(row):
    return len(row['text'].split(' '))

wiki['length'] = wiki.apply(compute_length) 


# In[40]:

nearest_neighbors_euclidean = model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
nearest_neighbors_euclidean = nearest_neighbors_euclidean.join(wiki[['name', 'length']], on={'reference_label':'name'})


# In[41]:

nearest_neighbors_euclidean.sort('rank')


# To see how these document lengths compare to the lengths of other documents in the corpus, let's make a histogram of the document lengths of Obama's 100 nearest neighbors and compare to a histogram of document lengths for all documents.

# In[42]:

plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])

plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()


# Relative to the rest of Wikipedia, nearest neighbors of Obama are overwhemingly short, most of them being shorter than 300 words. The bias towards short articles is not appropriate in this application as there is really no reason to  favor short articles over long articles (they are all Wikipedia articles, after all). Many of the Wikipedia articles are 300 words or more, and both Obama and Biden are over 300 words long.
# 
# **Note**: For the interest of computation time, the dataset given here contains _excerpts_ of the articles rather than full text. For instance, the actual Wikipedia article about Obama is around 25000 words. Do not be surprised by the low numbers shown in the histogram.

# **Note:** Both word-count features and TF-IDF are proportional to word frequencies. While TF-IDF penalizes very common words, longer articles tend to have longer TF-IDF vectors simply because they have more words in them.

# To remove this bias, we turn to **cosine distances**:
# $$
# d(\mathbf{x},\mathbf{y}) = 1 - \frac{\mathbf{x}^T\mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
# $$
# Cosine distances let us compare word distributions of two articles of varying lengths.
# 
# Let us train a new nearest neighbor model, this time with cosine distances.  We then repeat the search for Obama's 100 nearest neighbors.

# In[43]:

model2_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                  method='brute_force', distance='cosine')


# In[44]:

nearest_neighbors_cosine = model2_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
nearest_neighbors_cosine = nearest_neighbors_cosine.join(wiki[['name', 'length']], on={'reference_label':'name'})


# In[45]:

nearest_neighbors_cosine.sort('rank')


# From a glance at the above table, things look better.  For example, we now see Joe Biden as Barack Obama's nearest neighbor!  We also see Hillary Clinton on the list.  This list looks even more plausible as nearest neighbors of Barack Obama.
# 
# Let's make a plot to better visualize the effect of having used cosine distance in place of Euclidean on our TF-IDF vectors.

# In[46]:

plt.figure(figsize=(10.5,4.5))
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()


# Indeed, the 100 nearest neighbors using cosine distance provide a sampling across the range of document lengths, rather than just short articles like Euclidean distance provided.

# **Moral of the story**: In deciding the features and distance measures, check if they produce results that make sense for your particular application.

# # Problem with cosine distances: tweets vs. long articles

# Happily ever after? Not so fast. Cosine distances ignore all document lengths, which may be great in certain situations but not in others. For instance, consider the following (admittedly contrived) example.

# ```
# +--------------------------------------------------------+
# |                                             +--------+ |
# |  One that shall not be named                | Follow | |
# |  @username                                  +--------+ |
# |                                                        |
# |  Democratic governments control law in response to     |
# |  popular act.                                          |
# |                                                        |
# |  8:05 AM - 16 May 2016                                 |
# |                                                        |
# |  Reply   Retweet (1,332)   Like (300)                  |
# |                                                        |
# +--------------------------------------------------------+
# ```

# How similar is this tweet to Barack Obama's Wikipedia article? Let's transform the tweet into TF-IDF features, using an encoder fit to the Wikipedia dataset.  (That is, let's treat this tweet as an article in our Wikipedia dataset and see what happens.)

# In[47]:

sf = graphlab.SFrame({'text': ['democratic governments control law in response to popular act']})
sf['word_count'] = graphlab.text_analytics.count_words(sf['text'])

encoder = graphlab.feature_engineering.TFIDF(features=['word_count'], output_column_prefix='tf_idf')
encoder.fit(wiki)
sf = encoder.transform(sf)
sf


# Let's look at the TF-IDF vectors for this tweet and for Barack Obama's Wikipedia entry, just to visually see their differences.

# In[48]:

tweet_tf_idf = sf[0]['tf_idf.word_count']
tweet_tf_idf


# In[49]:

obama = wiki[wiki['name'] == 'Barack Obama']
obama


# Now, compute the cosine distance between the Barack Obama article and this tweet:

# In[50]:

obama_tf_idf = obama[0]['tf_idf']
graphlab.toolkits.distances.cosine(obama_tf_idf, tweet_tf_idf)


# Let's compare this distance to the distance between the Barack Obama article and all of its Wikipedia 10 nearest neighbors:

# In[51]:

model2_tf_idf.query(obama, label='name', k=10)


# With cosine distances, the tweet is "nearer" to Barack Obama than everyone else, except for Joe Biden!  This probably is not something we want. If someone is reading the Barack Obama Wikipedia page, would you want to recommend they read this tweet? Ignoring article lengths completely resulted in nonsensical results. In practice, it is common to enforce maximum or minimum document lengths. After all, when someone is reading a long article from _The Atlantic_, you wouldn't recommend him/her a tweet.
