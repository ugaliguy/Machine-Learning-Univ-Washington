
# coding: utf-8

# #Using deep features to build an image classifier
# 
# #Fire up GraphLab Create

# In[66]:

import ctypes, inspect, os, graphlab
from ctypes import wintypes
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,)
src_dir = os.path.split(inspect.getfile(graphlab))[0]
kernel32.SetDllDirectoryW(src_dir)


# In[68]:

import graphlab


# #Load a common image analysis dataset
# 
# We will use a popular benchmark dataset in computer vision called CIFAR-10.  
# 
# (We've reduced the data to just 4 categories = {'cat','bird','automobile','dog'}.)
# 
# This dataset is already split into a training set and test set.  

# In[69]:

image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')


# #Exploring the image data

# In[70]:

graphlab.canvas.set_target('ipynb')


# In[71]:

image_train['image'].show()


# #Train a classifier on the raw image pixels
# 
# We first start by training a classifier on just the raw pixels of the image.

# In[72]:

raw_pixel_model = graphlab.logistic_classifier.create(image_train,target='label',
                                              features=['image_array'])


# #Make a prediction with the simple model based on raw pixels

# In[73]:

image_test[0:3]['image'].show()


# In[74]:

image_test[0:3]['label']


# In[75]:

raw_pixel_model.predict(image_test[0:3])


# The model makes wrong predictions for all three images.

# #Evaluating raw pixel model on test data

# In[76]:

raw_pixel_model.evaluate(image_test)


# The accuracy of this model is poor, getting only about 46% accuracy.

# #Can we improve the model using deep features
# 
# We only have 2005 data points, so it is not possible to train a deep neural network effectively with so little data.  Instead, we will use transfer learning: using deep features trained on the full ImageNet dataset, we will train a simple model on this small dataset.

# In[77]:

len(image_train)


# ##Computing deep features for our images
# 
# The two lines below allow us to compute deep features.  This computation takes a little while, so we have already computed them and saved the results as a column in the data you loaded. 
# 
# (Note that if you would like to compute such deep features and have a GPU on your machine, you should use the GPU enabled GraphLab Create, which will be significantly faster for this task.)

# In[79]:

#deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
#image_train['deep_features'] = deep_learning_model.extract_features(image_train)


# As we can see, the column deep_features already contains the pre-computed deep features for this data. 

# In[80]:

image_train.head()


# #Given the deep features, let's train a classifier

# In[81]:

deep_features_model = graphlab.logistic_classifier.create(image_train,
                                                         features=['deep_features'],
                                                         target='label')


# #Apply the deep features model to first few images of test set

# In[82]:

image_test[0:3]['image'].show()


# In[83]:

deep_features_model.predict(image_test[0:3])


# The classifier with deep features gets all of these images right!

# #Compute test_data accuracy of deep_features_model
# 
# As we can see, deep features provide us with significantly better accuracy (about 78%)

# In[84]:

deep_features_model.evaluate(image_test)


# In[85]:

graphlab.canvas.set_target('browser')


# In[86]:

cat = image_test[0:1]


# In[87]:

image_train_cat = image_train[image_train['label'] == 'cat']


# In[88]:

knn_model_cat = graphlab.nearest_neighbors.create(image_train_cat,features=['deep_features'], label='id')


# In[89]:

nearest_cats = knn_model_cat.query(cat)


# In[90]:

nearest_cats.show()


# In[92]:

cat = image_test[0:1]
cat['image'].show()


# In[93]:

nearest_cats = knn_model_cat.query(cat)
nearest_cats


# In[94]:

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


# In[95]:

nearest_cats_images = get_images_from_ids(nearest_cats)


# In[96]:

nearest_cats_images['image'].show()


# In[97]:

image_train['label'].sketch_summary()


# In[98]:

image_train_dog = image_train[image_train['label'] == 'dog']


# In[99]:

knn_model_dog = graphlab.nearest_neighbors.create(image_train_dog, features=['deep_features'], label='id')


# In[100]:

nearest_dogs = knn_model_dog.query(cat)


# In[101]:

nearest_dogs_images = get_images_from_ids(nearest_dogs)


# In[102]:

nearest_dogs_images['image'].show()


# For the first image in the test data (image_test[0:1]), which we used above, compute the mean distance between this image at its 5 nearest neighbors that were labeled ‘cat’ in the training data. Do the same for dogs

# In[103]:

nearest_cats['distance'].mean()


# In[105]:

nearest_dogs['distance'].mean()


# In[106]:

image_test_dog = image_test[image_test['label'] == 'dog']
image_test_cat = image_test[image_test['label'] == 'cat']
image_test_automobile = image_test[image_test['label'] == 'automobile']
image_test_bird = image_test[image_test['label'] == 'bird']


# In[107]:

image_train_automobile = image_train[image_train['label'] == 'automobile']
image_train_bird = image_train[image_train['label'] == 'bird']


# In[108]:

knn_model_automobile = graphlab.nearest_neighbors.create(image_train_automobile,features=['deep_features'], label='id')
knn_model_bird = graphlab.nearest_neighbors.create(image_train_bird,features=['deep_features'], label='id')


# In[109]:

dog_dog_neighbors = knn_model_dog.query(image_test_dog, k=1)
dog_cat_neighbors = knn_model_cat.query(image_test_dog, k=1)
dog_automobile_neighbors = knn_model_automobile.query(image_test_dog, k=1)
dog_bird_neighbors = knn_model_bird.query(image_test_dog, k=1)


# In[111]:

dog_distances = graphlab.SFrame({'dog-dog': dog_dog_neighbors['distance'],
                                 'dog-cat': dog_cat_neighbors['distance'], 
                                 'dog-automobile': dog_automobile_neighbors['distance'], 
                                 'dog-bird': dog_bird_neighbors['distance']})


# In[112]:

print dog_distances


#  ###Consider one row of the SFrame dog_distances. Let’s call this variable row. 

# In[114]:

row = dog_distances[0]


# In[115]:

row['dog-cat']


# In[118]:

def is_dog_correct(row):
    if row['dog-dog'] < row['dog-cat'] and row['dog-dog'] < row['dog-automobile'] and row['dog-dog'] < row['dog-bird']:
        return 1
    else:
        return 0


# ###Computing the number of correct predictions for ‘dog’:

# In[119]:

correct_predictions = dog_distances.apply(is_dog_correct)


# In[120]:

total_correct_predictions = correct_predictions.sum()


# In[122]:

accuracy = float(total_correct_predictions) / len(dog_distances)


# In[123]:

print accuracy


# In[ ]:



