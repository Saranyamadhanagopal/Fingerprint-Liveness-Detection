#!/usr/bin/env python
# coding: utf-8

# #### Importing Packages

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

    ## This is the augmentation configuration we use for Training data
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(    
          rescale=1./255.,
          validation_split = 0.30
    )

    ###This is the augmentation configuration we use for Testing
    ## only rescaling. Other transformations are not required for Test data!!

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    ####This is a generator that will read pictures found in subfolders of data/train and
    ##indefinitely generate batches of augmented image data

    train_generator = train_datagen.flow_from_directory(
            '/home/iotadmin/database/Master_Train_Data', ##this is target directory
            target_size=(120, 120), ### all images will be resized to 150*150
            batch_size=batch_size,
            class_mode='categorical',
            subset = 'training'
            )

    ### this is a similar generator, for validation data
    validation_generator = train_datagen.flow_from_directory(
            '/home/iotadmin/database/Master_Train_Data/',
            target_size=(120, 120),
            batch_size=batch_size,
            class_mode='categorical',
            subset = 'validation')
    
    print('Compiling Model')
    
    # clear the current tensorflow graph and create new one
    tf.keras.backend.clear_session()

    #call the inception resnet pretrained model
    model_resnet = tf.keras.applications.InceptionResNetV2(weights='imagenet',include_top=False, input_shape=(120,120,3))
    last_layer = model_resnet.output
    
    #freeze the weights of the model
    for layer in model_resnet.layers:
        layer.trainable = False
       
    ### Flatten the last layer
    x = tf.keras.layers.Flatten()(last_layer)
    #x = x(last_layer)
    x = tf.keras.layers.Dropout(0.5)(x)

    # add fully-connected & dropout layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)   ##try with larger number of neurons 
    x = tf.keras.layers.Dropout(0.5)(x)
    n_classes=train_generator.num_classes

    ###output layer  - softmax layer for 2 classes
    out_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    # this is the model we will train
    Model = tf.keras.Model(inputs=model_resnet.input, outputs=out_layer)
    
    Model.compile(loss='binary_crossentropy', optimizer= 'Adam', metrics=['accuracy'])
    
    # checkpoint
    filepath="weights.adam.transfer1.preprocess.best.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    ####Tensorboard log
    log_dir ='./tf-log/crossmatch'
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    return Model,train_generator,test_datagen,validation_generator,checkpoint, tb_cb


# In[8]:


def train_resnet_model(train_path='/home/iotadmin/database/Precision samples/Precision data_Aug', batch_size = 128, 
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


# In[10]:


def classify(bestModel, img):
    model = tf.keras.models.load_model(bestModel)
    input_img = tf.keras.preprocessing.image.load_img(img,target_size=(120,120))
    input_img_arr = tf.keras.preprocessing.image.img_to_array(input_img)/255
    x=input_img_arr.reshape(1,120,120,3) ###numpy.expand_dims
    plt.imshow(input_img)
    result = model.predict(x)
    print("Probabilities: Live Spoof", result)


# In[7]:


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
    print('please check results_ResNet.csv in the path')

