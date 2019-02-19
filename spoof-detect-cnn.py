#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Import necessary packages
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam, SGD, Adadelta, RMSprop
tf.reset_default_graph()


# In[2]:


def model_parameters_optimisation(train_path,batch_size,best_model):
    
    tf.set_random_seed(12)
    print('Setting up ImageDataGenerator')
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(    
      rescale=1./255.,
      validation_split = 0.30)
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
                        train_path, ##this is target directory
                        target_size=(120, 120), ### all images will be resized to 150*150
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset = 'training')
    
    validation_generator = train_datagen.flow_from_directory(
                        train_path,
                        target_size=(120, 120),
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset = 'validation')
    
    print('Compiling Model')
    
    Model = Sequential()
    Model.add(Conv2D(32, (3, 3), input_shape=(120, 120, 3)))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))

    Model.add(Conv2D(64, (3, 3)))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))

    Model.add(Conv2D(64, (3, 3)))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Dropout(0.2))

    Model.add(Conv2D(128, (3, 3)))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))

    Model.add(Conv2D(128, (3, 3)))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Dropout(0.5))

    Model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    Model.add(Dense(512))  ####Fully connected layers
    Model.add(Activation('relu'))
    Model.add(Dropout(0.5))

    Model.add(Dense(2))
    Model.add(Activation('softmax'))

    #Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.0, amsgrad =False)
    #SGD = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    #keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
 
    Model.compile(loss='binary_crossentropy', optimizer= 'Adam', metrics=['accuracy'])
    
    filepath=best_model #### best model 
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    log_dir ='./tf-log/precision'
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    return Model,train_generator,test_datagen,validation_generator,checkpoint, tb_cb


# In[3]:


# train_model function will not return any values instead it automatically saves the model to the given weights path
def train_model(train_path='/home/iotadmin/database/Precision samples/Precision data_Aug', batch_size = 8, 
                best_model="weights.best.hdf5"):
    
    Model,train_generator,test_datagen,validation_generator,checkpoint,tb_cb = model_parameters_optimisation(train_path,
                                                                                                batch_size,best_model)
    
    print('Training model')
    
    fitted_model = Model.fit_generator(
                        train_generator,
                        steps_per_epoch= int(train_generator.samples) // batch_size,
                        epochs=40,
                        validation_data=validation_generator,
                        validation_steps= int(validation_generator.samples) // batch_size,
                        callbacks=[tb_cb,checkpoint])
    
    
    print('Labels:',train_generator.class_indices)
    print("Accuracy and Loss plots")
    
    # Training Accuracy
    plt.figure(figsize=(10, 6))  
    plt.plot(fitted_model.history['acc'])  
    #plt.plot(fitted_model_SGD.history['acc']) 
    #plt.plot(fitted_model_Adadelta.history['acc'])
    #plt.plot(fitted_model_RMSprop.history['acc'])
    plt.title('train. accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    #plt.legend(['adam', 'SGD', 'Adadelta','RMSprop'], loc='lower right')  
    plt.show()
    
    ### Validation accuracy
    plt.figure(figsize=(10, 6))  
    plt.plot(fitted_model.history['val_acc'])  
    #plt.plot(fitted_model_SGD.history['val_acc']) 
    #plt.plot(fitted_model_Adadelta.history['val_acc'])
    #plt.plot(fitted_model_RMSprop.history['val_acc'])
    plt.title('validataion accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    #plt.legend(['adam', 'SGD', 'Adadelta','RMSprop'], loc='lower right')  
    plt.show()
    
    ### Training loss
    plt.figure(figsize=(10, 6))  
    plt.plot(fitted_model.history['loss'])  
    #plt.plot(fitted_model_SGD.history['loss']) 
    #plt.plot(fitted_model_Adadelta.history['loss'])
    #plt.plot(fitted_model_RMSprop.history['loss'])
    plt.title('train loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    #plt.legend(['adam', 'SGD', 'Adadelta','RMSprop'], loc='lower right')  
    plt.show()
    
    ### Validation loss
    plt.figure(figsize=(10, 6))  
    plt.plot(fitted_model.history['val_loss'])  
    #plt.plot(fitted_model_SGD.history['val_loss']) 
    #plt.plot(fitted_model_Adadelta.history['val_loss'])
    #plt.plot(fitted_model_RMSprop.history['val_loss'])
    plt.title('validation loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    #plt.legend(['adam', 'SGD', 'Adadelta','RMSprop'], loc='lower right')  
    plt.show()
    print('Best model is saved')
    print('Model is ready to predict')


# #### Predicting Single Image

# In[27]:


def classify(bestModel, img):
    model = tf.keras.models.load_model(bestModel)
    input_img = tf.keras.preprocessing.image.load_img(img,target_size=(120,120))
    input_img_arr = tf.keras.preprocessing.image.img_to_array(input_img)/255
    x=input_img_arr.reshape(1,120,120,3) ###numpy.expand_dims
    plt.imshow(input_img)
    result = model.predict(x)
    print("Probabilities: Live Spoof", result)


# #### Predicting multiple images

# In[29]:


def predict(Path, best_model, batch_size):
    
    print('Loading Best Model')
    model = tf.keras.models.load_model(best_model)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
                        Path,  # this is the target directory
                        target_size=(120, 120),  # all images will be resized to 150x150
                        batch_size=batch_size,
                        shuffle = False,
                        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary label
    test_generator.reset()
    
    print('Predicting Model')
    
    # this returns the probabilities
    test_prob = model.predict_generator(test_generator, 1) 
    print('Probabilities: Live Spoof',test_prob)
    
    # convert probabilities to classes
    test_pred_classes = np.argmax(test_prob, axis=1) 
    #print('Classes: Live 0 Spoof 1',test_pred_classes)
    
    # Check the corresponding filenames of the predictions
    #print(test_generator.filenames)
    
    #Save predictions
    test_predictions=pd.DataFrame({"Filename":test_generator.filenames, "Predictions":test_pred_classes})
    test_predictions.to_csv('results_CNN.csv', header=True, sep=',')
    print('please check results_CNN.csv in the path')

