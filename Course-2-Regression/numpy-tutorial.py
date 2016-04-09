
# coding: utf-8

# #Numpy Tutorial

# Numpy is a computational library for Python that is optimized for operations on multi-dimensional arrays. In this notebook we will use numpy to work with 1-d arrays (often called vectors) and 2-d arrays (often called matrices).
# 
# For a the full user guide and reference for numpy see: http://docs.scipy.org/doc/numpy/

# In[1]:

import numpy as np # importing this way allows us to refer to numpy as np


# # Creating Numpy Arrays

# New arrays can be made in several ways. We can take an existing list and convert it to a numpy array:

# In[7]:

mylist = [1., 2., 3., 4.]
mynparray = np.array(mylist)
mynparray


# You can initialize an array (of any dimension) of all ones or all zeroes with the ones() and zeros() functions:

# In[8]:

one_vector = np.ones(4)
print one_vector # using print removes the array() portion


# In[12]:

one2Darray = np.ones((2, 4)) # an 2D array with 2 "rows" and 4 "columns"
print one2Darray


# In[13]:

zero_vector = np.zeros(4)
print zero_vector


# You can also initialize an empty array which will be filled with values. This is the fastest way to initialize a fixed-size numpy array however you must ensure that you replace all of the values.

# In[14]:

empty_vector = np.empty(5)
print empty_vector


# #Accessing array elements

# Accessing an array is straight forward. For vectors you access the index by referring to it inside square brackets. Recall that indices in Python start with 0.

# In[15]:

mynparray[2]


# 2D arrays are accessed similarly by referring to the row and column index separated by a comma:

# In[20]:

my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
print my_matrix


# In[21]:

print my_matrix[1, 2]


# Sequences of indices can be accessed using ':' for example

# In[22]:

print my_matrix[0:2, 2] # recall 0:2 = [0, 1]


# In[23]:

print my_matrix[0, 0:3]


# You can also pass a list of indices. 

# In[24]:

fib_indices = np.array([1, 1, 2, 3])
random_vector = np.random.random(10) # 10 random numbers between 0 and 1
print random_vector


# In[25]:

print random_vector[fib_indices]


# You can also use true/false values to select values

# In[28]:

my_vector = np.array([1, 2, 3, 4])
select_index = np.array([True, False, True, False])
print my_vector[select_index]


# For 2D arrays you can select specific columns and specific rows. Passing ':' selects all rows/columns

# In[29]:

select_cols = np.array([True, False, True]) # 1st and 3rd column
select_rows = np.array([False, True]) # 2nd row


# In[30]:

print my_matrix[select_rows, :] # just 2nd row but all columns


# In[31]:

print my_matrix[:, select_cols] # all rows and just the 1st and 3rd column


# #Operations on Arrays

# You can use the operations '\*', '\*\*', '\\', '+' and '-' on numpy arrays and they operate elementwise.

# In[33]:

my_array = np.array([1., 2., 3., 4.])
print my_array*my_array


# In[34]:

print my_array**2


# In[35]:

print my_array - np.ones(4)


# In[36]:

print my_array + np.ones(4)


# In[37]:

print my_array / 3


# In[38]:

print my_array / np.array([2., 3., 4., 5.]) # = [1.0/2.0, 2.0/3.0, 3.0/4.0, 4.0/5.0]


# You can compute the sum with np.sum() and the average with np.average()

# In[39]:

print np.sum(my_array)


# In[40]:

print np.average(my_array)


# In[41]:

print np.sum(my_array)/len(my_array)


# #The dot product

# An important mathematical operation in linear algebra is the dot product. 
# 
# When we compute the dot product between two vectors we are simply multiplying them elementwise and adding them up. In numpy you can do this with np.dot()

# In[42]:

array1 = np.array([1., 2., 3., 4.])
array2 = np.array([2., 3., 4., 5.])
print np.dot(array1, array2)


# In[43]:

print np.sum(array1*array2)


# Recall that the Euclidean length (or magnitude) of a vector is the squareroot of the sum of the squares of the components. This is just the squareroot of the dot product of the vector with itself:

# In[44]:

array1_mag = np.sqrt(np.dot(array1, array1))
print array1_mag


# In[46]:

print np.sqrt(np.sum(array1*array1))


# We can also use the dot product when we have a 2D array (or matrix). When you have an vector with the same number of elements as the matrix (2D array) has columns you can right-multiply the matrix by the vector to get another vector with the same number of elements as the matrix has rows. For example this is how you compute the predicted values given a matrix of features and an array of weights.

# In[47]:

my_features = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
print my_features


# In[48]:

my_weights = np.array([0.4, 0.5])
print my_weights


# In[49]:

my_predictions = np.dot(my_features, my_weights) # note that the weights are on the right
print my_predictions # which has 4 elements since my_features has 4 rows


# Similarly if you have a vector with the same number of elements as the matrix has *rows* you can left multiply them.

# In[50]:

my_matrix = my_features
my_array = np.array([0.3, 0.4, 0.5, 0.6])


# In[51]:

print np.dot(my_array, my_matrix) # which has 2 elements because my_matrix has 2 columns


# #Multiplying Matrices

# If we have two 2D arrays (matrices) matrix_1 and matrix_2 where the number of columns of matrix_1 is the same as the number of rows of matrix_2 then we can use np.dot() to perform matrix multiplication.

# In[52]:

matrix_1 = np.array([[1., 2., 3.],[4., 5., 6.]])
print matrix_1


# In[53]:

matrix_2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
print matrix_2


# In[54]:

print np.dot(matrix_1, matrix_2)


# In[ ]:



