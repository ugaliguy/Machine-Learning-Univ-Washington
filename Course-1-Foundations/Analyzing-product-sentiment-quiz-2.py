
# coding: utf-8

# #Predicting sentiment from product reviews
# 
# #Fire up GraphLab Create

# In[3]:

import graphlab


# #Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[4]:

products = graphlab.SFrame('amazon_baby.gl/')


# #Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[5]:

products.head()


# #Build the word count vector for each review

# In[5]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[6]:

products.head()


# In[7]:

graphlab.canvas.set_target('ipynb')


# In[8]:

products['name'].show()


# # Word count vector for each review

# In[6]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])
products.head()


# #Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[8]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[9]:

len(giraffe_reviews)


# In[10]:

giraffe_reviews['rating'].show(view='Categorical')


# #Build a sentiment classifier

# In[11]:

products['rating'].show(view='Categorical')


# ##Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[12]:

#ignore all 3* reviews
products = products[products['rating'] != 3]


# In[13]:

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4
products['sentiment'].show('Categorical')


# In[14]:

products.head()


# ##Let's train the sentiment classifier

# In[15]:

train_data,test_data = products.random_split(.8, seed=0)
test_data['sentiment'].show('Categorical')


# In[17]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)
print sentiment_model['coefficients'].sort('value', ascending=False).print_rows(num_rows=15, num_columns=10)


# #Evaluate the sentiment model

# In[18]:

print "Evaluation of sentiment_model ..."
sentiment_model.evaluate(test_data, metric='roc_curve')


# In[21]:

graphlab.canvas.set_target('ipynb')
sentiment_model.show(view='Evaluation')


# In[22]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

for word in selected_words:
    products[word] = products['word_count'].apply(lambda word_count_dic: word_count_dic[word] if word in word_count_dic.keys() 
                                                 else 0, dtype = int)
    print 'The number of instances of \"%s\" is %s.' % (word, products[word].sum())

graphlab.canvas.set_target('ipynb')    
products.show()


# In[25]:

train_data,test_data = products.random_split(.8, seed=0)


# In[26]:

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)
print selected_words_model['coefficients'].sort('value', ascending=False).print_rows(num_rows=15, num_columns=10)
graphlab.canvas.set_target('ipynb')    
products.show()


# In[27]:

selected_words_model.evaluate(test_data, metric='roc_curve')
selected_words_model.show(view='Evaluation')
graphlab.canvas.set_target('ipynb')    
products.show()


# #Applying the learned model to understand sentiment for Baby Trend Diaper Champ

# In[28]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[30]:

diaper_champ_reviews['predicted_sentiment'] = selected_words_model.predict(diaper_champ_reviews, output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('rating', ascending=False)


# In[31]:

print selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')


# In[32]:

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('rating', ascending=False)
print selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')


# In[33]:

for word in selected_words:
    print diaper_champ_reviews[0][word]


# In[ ]:



