from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import load_model

######################### Step1:  Initialize the parameters and file paths #########################
# configure the parameters
batch_size = 20
num_classes = 4
epochs = 50
image_height = 384
image_width = 512

# Set the corresponding file paths
model_folder = 'BCNN'
# Configure the train, val, and test p
base_dir = 'D:/Data/Image/Biomedicine/integrated/breasthistology/Reinhard_Patches_CNN'
train_dir = os.path.join(base_dir, 'Training_data')
validation_dir = os.path.join(base_dir, 'Validation_data')
# test_dir = os.path.join(base_dir, 'Test_data/Initial')
# test_dir = os.path.join(base_dir, 'Test_data/Extended')
test_dir = os.path.join(base_dir, 'Test_data/Overall')

######################### Step2:  Obtain the test dataflow #########################
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_height, image_width),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')

######################### Step3:  Load the best model trained before and evaluate its performance #########################
# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
# load the best model
# If the error appears 'Error in loading the saved optimizer', it doesn't matter!
best_model = load_model(os.path.join(checkpoint_dir, "weights-improvement-28-0.78.hdf5"))

######################### Step3.1:  Obtain the loss and acc
# Score trained best_model.
print('The evaluation starts!\n')
scores = best_model.evaluate_generator(
    test_generator,
    steps=len(test_generator))

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

######################### Step3.2:  Obtain the precision 、recall 、f1-score
# Obtain the prediction
pred_prob = best_model.predict_generator(
    test_generator,
    steps=len(test_generator))

# The index for prediction of testing set
pred_labels = np.argmax(pred_prob, axis=1)
# The scores for prediction of testing set
pred_scores = np.amax(pred_prob, axis=1)

# The true labels of testing set
true_labels = test_generator.classes

# Print classification results
cfm = confusion_matrix(true_labels, pred_labels)
print(classification_report(true_labels, pred_labels, digits=4))









