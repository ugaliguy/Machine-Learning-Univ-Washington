
# coding: utf-8

# # Regression Week 4: Ridge Regression (interpretation)

# In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
# * Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression
# * Use matplotlib to visualize polynomial regressions
# * Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression, this time with L2 penalty
# * Use matplotlib to visualize polynomial regressions under L2 regularization
# * Choose best L2 penalty using cross-validation.
# * Assess the final fit using test data.
# 
# We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

# # Fire up graphlab create

# In[1]:

import graphlab


# # Polynomial regression, revisited

# We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:

# In[3]:

def polynomial_sframe(feature, degree):
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


# Let's use matplotlib to visualize what a polynomial regression looks like on the house data.

# In[4]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[5]:

sales = graphlab.SFrame('kc_house_data.gl/')


# As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.

# In[6]:

sales = sales.sort(['sqft_living','price'])


# Let us revisit the 15th-order polynomial model using the 'sqft_living' input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. When fitting the model, use an L2 penalty of `1e-5`:

# In[11]:

l2_small_penalty = 1e-5
poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features15 = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target


# Note: When we have so many features and so few data points, the solution can become highly numerically unstable, which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will introduce a tiny amount of regularization (`l2_penalty=1e-5`) to make the solution numerically stable.  (In lecture, we discussed the fact that regularization can also help with numerical stability, and here we are seeing a practical example.)
# 
# With the L2 penalty specified above, fit the model and print out the learned weights.
# 
# Hint: make sure to add 'price' column to the new SFrame before calling `graphlab.linear_regression.create()`. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set=None` in this call.

# In[14]:

model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features15, l2_penalty=l2_small_penalty, 
                                            validation_set = None)
model15.get("coefficients").print_rows(num_rows = 16)


# ***QUIZ QUESTION:  What's the learned value for the coefficient of feature `power_1`?***

# # Observe overfitting

# Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. But first, we must reproduce the experiment we did in Week 3.

# First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use `.random_split` function and make sure you set `seed=0`. 

# In[15]:

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)


# Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.
# 
# Hint: When calling `graphlab.linear_regression.create()`, use the same L2 penalty as before (i.e. `l2_small_penalty`).  Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.

# In[23]:

poly1_data = polynomial_sframe(set_1['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly1_data.column_names() 
poly1_data['price'] = set_1['price']
model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_small_penalty)
model1.get("coefficients").print_rows(num_rows=16)
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')


# In[25]:

poly2_data = polynomial_sframe(set_2['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly2_data.column_names() 
poly2_data['price'] = set_2['price']
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_small_penalty)
model2.get("coefficients").print_rows(num_rows=16)
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')


# In[26]:

poly3_data = polynomial_sframe(set_3['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly3_data.column_names() 
poly3_data['price'] = set_3['price']
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_small_penalty)
model3.get("coefficients").print_rows(num_rows=16)
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')


# In[27]:

poly4_data = polynomial_sframe(set_4['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly4_data.column_names() 
poly4_data['price'] = set_4['price']
model4 = graphlab.linear_regression.create(poly4_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_small_penalty)
model4.get("coefficients").print_rows(num_rows=16)
plt.plot(poly4_data['power_1'],poly4_data['price'],'.',
        poly4_data['power_1'], model4.predict(poly4_data),'-')


# The four curves should differ from one another a lot, as should the coefficients you learned.
# 
# ***QUIZ QUESTION:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?***  (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# # Ridge regression comes to rescue

# Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)
# 
# With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.

# In[30]:

l2_large_penalty=1e5


# In[31]:

poly1_data = polynomial_sframe(set_1['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly1_data.column_names() 
poly1_data['price'] = set_1['price']
model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_large_penalty)
model1.get("coefficients").print_rows(num_rows=16)
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')


# In[32]:

poly2_data = polynomial_sframe(set_2['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly2_data.column_names() 
poly2_data['price'] = set_2['price']
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_large_penalty)
model2.get("coefficients").print_rows(num_rows=16)
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')


# In[33]:

poly3_data = polynomial_sframe(set_3['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly3_data.column_names() 
poly3_data['price'] = set_3['price']
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_large_penalty)
model3.get("coefficients").print_rows(num_rows=16)
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')


# In[34]:

poly4_data = polynomial_sframe(set_4['sqft_living'], 15) # use equivalent of `polynomial_sframe`
my_features = poly4_data.column_names() 
poly4_data['price'] = set_4['price']
model4 = graphlab.linear_regression.create(poly4_data, target = 'price', features = my_features, verbose = False, 
                                           validation_set = None, l2_penalty=l2_large_penalty)
model4.get("coefficients").print_rows(num_rows=16)
plt.plot(poly4_data['power_1'],poly4_data['price'],'.',
        poly4_data['power_1'], model4.predict(poly4_data),'-')


# These curves should vary a lot less, now that you applied a high degree of regularization.
# 
# ***QUIZ QUESTION:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?*** (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# # Selecting an L2 penalty via cross-validation

# Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.
# 
# We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:
# 
# Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
# Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
# ...<br>
# Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# 
# After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 
# 
# To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. GraphLab Create has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder. (Make sure to use `seed=1` to get consistent answer.)

# In[35]:

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)


# Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

# With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.

# In[36]:

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)


# Let us familiarize ourselves with array slicing with SFrame. To extract a continuous slice from an SFrame, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.

# In[37]:

train_valid_shuffled[0:10] # rows 0 to 9


# Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
# Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.

# In[40]:

validation4 = train_valid_shuffled[5818: 7758]


# To verify that we have the right elements extracted, run the following cell, which computes the average price of the fourth segment. When rounded to nearest whole number, the average should be $536,234.

# In[41]:

print int(round(validation4['price'].mean(), 0))


# After designating one of the k segments as the validation set, we train a model using the rest of the data. To choose the remainder, we slice (0:start) and (end+1:n) of the data and paste them together. SFrame has `append()` method that pastes together two disjoint sets of rows originating from a common dataset. For instance, the following cell pastes together the first and last two rows of the `train_valid_shuffled` dataframe.

# In[42]:

n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print first_two.append(last_two)


# Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.

# In[43]:

n = len(train_valid_shuffled)
before_third = train_valid_shuffled[0:5818]
after_third = train_valid_shuffled[7758:n]
train4 = before_third.append(after_third) 


# To verify that we have the right elements extracted, run the following cell, which computes the average price of the data with fourth segment excluded. When rounded to nearest whole number, the average should be $539,450.

# In[44]:

print int(round(train4['price'].mean(), 0))


# Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.
# 
# * For each i in [0, 1, ..., k-1]:
#   * Compute starting and ending indices of segment i and call 'start' and 'end'
#   * Form validation set by taking a slice (start:end+1) from the data.
#   * Form training set by appending slice (end+1:n) to the end of slice (0:start).
#   * Train a linear model using training set just formed, with a given l2_penalty
#   * Compute validation error using validation set just formed

# In[45]:

def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    n = len(data)
    errors = []
    for i in range(0, k):
        start = (n*i)/k
        end = (n*(i+1)/k)-1
        validation_set = data[start:end+1]
        training_set = data[0:start].append(data[end+1:n])
        
        model = graphlab.linear_regression.create(training_set,
                                                  target = output_name, features = features_list,
                                                  l2_penalty=l2_penalty,
                                                  validation_set = None,
                                                  verbose = False)
        
        price_validation_predicted = model.predict(validation_set)
        val_errors = price_validation_predicted - validation_set['price']
        RSS = sum(val_errors * val_errors)
        errors.append(RSS)
        
    return sum(errors)/len(errors)


# Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
# * We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
# * For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
#     * Run 10-fold cross-validation with `l2_penalty`
# * Report which L2 penalty produced the lowest average validation error.
# 
# Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!

# In[48]:

import numpy

data15 = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
set15_features = data15.column_names()
data15['price'] = train_valid_shuffled['price']

min_error = None
best_l2_penalty = None
cross_val_errors = []
l2_penalty_values = numpy.logspace(1, 7, num=13)

for l2_penalty in l2_penalty_values:
    print l2_penalty
    avg_val_error = k_fold_cross_validation(10, l2_penalty, data15, 'price', set15_features)
    print avg_val_error
    cross_val_errors.append(avg_val_error)
    if min_error is None or avg_val_error < min_error:
        min_error = avg_val_error
        best_l2_penalty = l2_penalty

print "Best : " + str(best_l2_penalty)


# ***QUIZ QUESTIONS:  What is the best value for the L2 penalty according to 10-fold validation?***

# You may find it useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  

# In[49]:

# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.
plt.plot(l2_penalty_values,cross_val_errors,'k-')
plt.xlabel('$\L2_penalty$')
plt.ylabel('K-fold cross validation error')
plt.xscale('log')
plt.yscale('log')


# Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of `l2_penalty`.  This way, your final model will be trained on the entire dataset.

# In[50]:

train_data = polynomial_sframe(sales['sqft_living'], 15)
train_features = train_data.column_names()
train_data['price'] = sales['price']

train_model = graphlab.linear_regression.create(train_data, 
                                               target = 'price', features = train_features,
                                               l2_penalty=best_l2_penalty,
                                               validation_set = None)

train_model.get("coefficients").print_rows(num_rows = 16)

plt.plot(train_data['power_1'],train_data['price'],'.',
        train_data['power_1'], train_model.predict(train_data),'-')


# ***QUIZ QUESTION: Using the best L2 penalty found above, train a model using all training data. What is the RSS on the TEST data of the model you learn with this L2 penalty? ***

# In[51]:

test_data = polynomial_sframe(test['sqft_living'], 15)
predictions_test = train_model.predict(test_data)
test_errors = predictions_test - test['price']
RSS_test = sum(test_errors * test_errors)
print RSS_test


# In[ ]:



