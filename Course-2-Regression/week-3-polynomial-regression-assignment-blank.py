
# coding: utf-8

# # Regression Week 3: Assessing Fit (polynomial regression)

# In this notebook you will compare different regression models in order to assess which model fits best. We will be using polynomial regression as a means to examine this topic. In particular you will:
# * Write a function to take an SArray and a degree and return an SFrame where each column is the SArray to a polynomial value up to the total degree e.g. degree = 3 then column 1 is the SArray column 2 is the SArray squared and column 3 is the SArray cubed
# * Use matplotlib to visualize polynomial regressions
# * Use matplotlib to visualize the same polynomial degree on different subsets of the data
# * Use a validation set to select a polynomial degree
# * Assess the final fit using test data
# 
# We will continue to use the House data from previous notebooks.

# # Fire up graphlab create

# In[1]:

import graphlab


# Next we're going to write a polynomial function that takes an SArray and a maximal degree and returns an SFrame with columns containing the SArray to all the powers up to the maximal degree.
# 
# The easiest way to apply a power to an SArray is to use the .apply() and lambda x: functions. 
# For example to take the example array and compute the third power we can do as follows: (note running this cell the first time may take longer than expected since it loads graphlab)

# In[2]:

tmp = graphlab.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print tmp
print tmp_cubed


# We can create an empty SFrame using graphlab.SFrame() and then add any columns to it with ex_sframe['column_name'] = value. For example we create an empty SFrame and make the column 'power_1' to be the first power of tmp (i.e. tmp itself).

# In[17]:

ex_sframe = graphlab.SFrame()
ex_sframe['power_1'] = tmp
print ex_sframe


# # Polynomial_sframe function

# Using the hints above complete the following function to create an SFrame consisting of the powers of an SArray up to a specific degree:

# In[23]:

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature.apply(lambda x: x**power)
    return poly_sframe


# To test your function consider the smaller tmp variable and what you would expect the outcome of the following call:

# In[24]:

print polynomial_sframe(tmp, 3)


# # Visualizing polynomial regression

# Let's use matplotlib to visualize what a polynomial regression looks like on some real data.

# In[25]:

sales = graphlab.SFrame('kc_house_data.gl/')


# As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.

# In[27]:

sales = sales.sort(['sqft_living', 'price'])


# Let's start with a degree 1 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.

# In[28]:

poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target


# NOTE: for all the models in this notebook use validation_set = None to ensure that all results are consistent across users.

# In[29]:

model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)


# In[30]:

#let's take a look at the weights before we plot
model1.get("coefficients")


# In[31]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[32]:

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')


# Let's unpack that plt.plot() command. The first pair of SArrays we passed are the 1st power of sqft and the actual price we then ask it to print these as dots '.'. The next pair we pass is the 1st power of sqft and the predicted values from the linear model. We ask these to be plotted as a line '-'. 
# 
# We can see, not surprisingly, that the predicted values all fall on a line, specifically the one with slope 280 and intercept -43579. What if we wanted to plot a second degree polynomial?

# In[33]:

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)


# In[34]:

model2.get("coefficients")


# In[35]:

plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')


# The resulting model looks like half a parabola. Try on your own to see what the cubic looks like:

# In[36]:

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features3 = poly3_data.column_names() # get the name of the features
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features3, validation_set = None)


# In[38]:

model3.get("coefficients")


# In[39]:

plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')


# Now try a 15th degree polynomial:

# In[40]:

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features15 = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features15, validation_set = None)


# In[41]:

model15.get("coefficients")


# In[42]:

plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
        poly15_data['power_1'], model15.predict(poly15_data),'-')


# What do you think of the 15th degree polynomial? Do you think this is appropriate? If we were to change the data do you think you'd get pretty much the same curve? Let's take a look.

# # Changing the data and re-learning

# We're going to split the sales data into four subsets of roughly equal size. Then you will estimate a 15th degree polynomial model on all four subsets of the data. Print the coefficients (you should use .print_rows(num_rows = 16) to view all of them) and plot the resulting fit (as we did above). The quiz will ask you some questions about these results.
# 
# To split the sales data into four subsets, we perform the following steps:
# * First split sales into 2 subsets with `.random_split(0.5, seed=0)`. 
# * Next split the resulting subsets into 2 more subsets each. Use `.random_split(0.5, seed=0)`.
# 
# We set `seed=0` in these steps so that different users get consistent results.
# You should end up with 4 subsets (`set_1`, `set_2`, `set_3`, `set_4`) of approximately equal size. 

# In[45]:

set_11, set_22 = sales.random_split(0.5, seed=0)
set_1, set_2 = set_11.random_split(0.5, seed=0)
set_3, set_4 = set_22.random_split(0.5, seed=0)


# Fit a 15th degree polynomial on set_1, set_2, set_3, and set_4 using sqft_living to predict prices. Print the coefficients and make a plot of the resulting model.

# In[46]:

set_1_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features_1 = set_1_data.column_names() # get the name of the features
set_1_data['price'] = set_1['price'] # add price to the data since it's the target
model_1 = graphlab.linear_regression.create(set_1_data, target = 'price', features = my_features_1, validation_set = None)
model_1.get("coefficients").print_rows(num_rows = 16)
plt.plot(set_1_data['power_1'],set_1_data['price'],'.',set_1_data['power_1'], model_1.predict(set_1_data),'-')


# In[47]:

set_2_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features_2 = set_2_data.column_names() # get the name of the features
set_2_data['price'] = set_2['price'] # add price to the data since it's the target
model_2 = graphlab.linear_regression.create(set_1_data, target = 'price', features = my_features_2, validation_set = None)
model_2.get("coefficients").print_rows(num_rows = 16)
plt.plot(set_2_data['power_1'],set_2_data['price'],'.',set_2_data['power_1'], model_2.predict(set_2_data),'-')


# In[48]:

set_3_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features_3 = set_3_data.column_names() # get the name of the features
set_3_data['price'] = set_3['price'] # add price to the data since it's the target
model_3 = graphlab.linear_regression.create(set_3_data, target = 'price', features = my_features_3, validation_set = None)
model_3.get("coefficients").print_rows(num_rows = 16)
plt.plot(set_3_data['power_1'],set_3_data['price'],'.',set_3_data['power_3'], model_3.predict(set_3_data),'-')


# In[49]:

set_4_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features_4 = set_4_data.column_names() # get the name of the features
set_4_data['price'] = set_4['price'] # add price to the data since it's the target
model_4 = graphlab.linear_regression.create(set_4_data, target = 'price', features = my_features_4, validation_set = None)
model_4.get("coefficients").print_rows(num_rows = 16)
plt.plot(set_4_data['power_4'],set_4_data['price'],'.',set_4_data['power_4'], model_4.predict(set_4_data),'-')


# Some questions you will be asked on your quiz:
# 
# **Quiz Question: Is the sign (positive or negative) for power_15 the same in all four models?**   NO
# 
# **Quiz Question: (True/False) the plotted fitted lines look the same in all four plots** False

# # Selecting a Polynomial Degree

# Whenever we have a "magic" parameter like the degree of the polynomial there is one well-known way to select these parameters: validation set. (We will explore another approach in week 4).
# 
# We split the sales dataset 3-way into training set, test set, and validation set as follows:
# 
# * Split our sales data into 2 sets: `training_and_validation` and `testing`. Use `random_split(0.9, seed=1)`.
# * Further split our training data into two sets: `training` and `validation`. Use `random_split(0.5, seed=1)`.
# 
# Again, we set `seed=1` to obtain consistent results for different users.

# In[51]:

training_and_validation, testing = sales.random_split(0.9, seed=1)
training, validation = training_and_validation.random_split(0.5, seed=1)


# Next you should write a loop that does the following:
# * For degree in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (to get this in python type range(1, 15+1))
#     * Build an SFrame of polynomial data of train_data['sqft_living'] at the current degree
#     * hint: my_features = poly_data.column_names() gives you a list e.g. ['power_1', 'power_2', 'power_3'] which you might find useful for graphlab.linear_regression.create( features = my_features)
#     * Add train_data['price'] to the polynomial SFrame
#     * Learn a polynomial regression model to sqft vs price with that degree on TRAIN data
#     * Compute the RSS on VALIDATION data (here you will want to use .predict()) for that degree and you will need to make a polynmial SFrame using validation data.
# * Report which degree had the lowest RSS on validation data (remember python indexes from 0)
# 
# (Note you can turn off the print out of linear_regression.create() with verbose = False)

# In[52]:

for i in range(1,16):
    poly_data = polynomial_sframe(training['sqft_living'], i)
    my_features = poly_data.column_names() 
    poly_data['price'] = training['price'] # add price to the data since it's the target
    model = graphlab.linear_regression.create(poly_data, target = 'price', features = my_features, 
                                              validation_set = None, verbose = False)
    validation_data = polynomial_sframe(validation['sqft_living'], i)
    validation_data['price'] = validation['price']
    predictions = model.predict(validation_data)
    RSS = ((predictions - validation_data['price']) * (predictions - validation_data['price'])).sum()
    print str(RSS)


# **Quiz Question: Which degree (1, 2, …, 15) had the lowest RSS on Validation data?**

# Now that you have chosen the degree of your polynomial using validation data, compute the RSS of this model on TEST data. Report the RSS on your quiz.

# In[57]:

train_data = polynomial_sframe(training['sqft_living'], 6)
train_features = train_data.column_names() # get the name of the features
train_data['price'] = training['price'] # add price to the data since it's the target
model = graphlab.linear_regression.create(train_data, target = 'price', features = train_features, validation_set = None, verbose=False)

test_data = polynomial_sframe(testing['sqft_living'], 6)
test_data['price'] = testing['price'] # add price to the data since it's the target

# First get the predictions
predictions = model.predict(test_data)
# then compute the residuals (since we are squaring it doesn't matter which order you subtract)
residuals = test_data['price'] - predictions
# square the residuals and add them up
residuals_squared = residuals * residuals
RSS = residuals_squared.sum()
print("Degree: %s, Test Data RSS: $%.6f" % (6, RSS))


# **Quiz Question: what is the RSS on TEST data for the model with the degree selected from Validation data?**

# In[ ]:



