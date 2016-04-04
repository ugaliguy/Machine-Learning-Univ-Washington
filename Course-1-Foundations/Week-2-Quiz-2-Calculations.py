
# coding: utf-8

# ## Week 2  Quiz 2

# ### 1. Selection and summary statistics: 
# In the notebook we covered in the module, we discovered which neighborhood (zip code) of Seattle had the highest average house sale price. Now, take the sales data, select only the houses with this zip code, and compute the average price. Save this result to answer the quiz at the end.

# In[1]:

import graphlab


# In[2]:

sales = graphlab.SFrame('home_data.gl/')


# In[3]:

sales


# In[25]:

graphlab.canvas.set_target('ipynb')
sales.show(view="Box Plot", x="zipcode", y="price")


# ### First attempt at finding average price of houses in 'most expensive' zip code
sales.groupby("zipcode",{'average_price_by_zipcode':graphlab.aggregate.AVG('price')})
# In[14]:

print sales.groupby("zipcode",{'average_price_by_zipcode':graphlab.aggregate.AVG('price')})


# In[15]:

print sales.groupby("zipcode",{'average_price_by_zipcode':graphlab.aggregate.AVG('price')}).print_rows(num_rows = 70)


# In[26]:

highPriceZip = '98039'


# In[30]:

high_sales = sales[sales['zipcode'] == highPriceZip]


# In[31]:

print 'Average of sales prices in zipcode with highest average price: ', high_sales['price'].mean()


# ### 2.Filtering Data
# * Using such filters, first select the houses that have ‘sqft_living’ higher than 2000 sqft but no larger than 4000 sqft.
# * What fraction of the all houses have ‘sqft_living’ in this range?

# In[32]:

filtered_sqft_living = sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] < 4000) ]


# In[34]:

number_filtered = len(filtered_sqft_living)


# In[35]:

print number_filtered


# In[38]:

fraction = number_filtered/(len(sales['sqft_living'])*1.0)


# In[39]:

print fraction


# ### 3. Building a regression model with several more features:
# Compute the RMSE (root mean squared error) on the test_data for the model using just my_features, and for the one using advanced_features.

# In[40]:

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']


# In[42]:

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors 
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[44]:

train_data, test_data = sales.random_split(0.8, seed=0)


# In[45]:

my_features_model = graphlab.linear_regression.create(test_data, target='price', features=my_features, validation_set=None)


# In[46]:

advanced_features_model = graphlab.linear_regression.create(test_data, target='price', 
                                                            features=advanced_features, validation_set=None)


# In[ ]:



