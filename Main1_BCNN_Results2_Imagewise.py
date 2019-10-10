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
channels = 3
# Set the corresponding file paths
model_folder = 'BCNN'
# Configure the train, val, and test
base_dir = 'D:/Data/Image/Biomedicine/integrated/breasthistology/Reinhard_Patches_CNN'
# test_dir = os.path.join(base_dir, 'Test_data/Initial')
# test_dir= os.path.join(base_dir, 'Test_data/Extended')
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

######################### Step3.1:  Obtain the precision 、recall 、f1-score
NImages = len(test_generator.classes)//49 # 49 patches for each image

# Obtain the prediction
pred_prob = best_model.predict_generator(
    test_generator,
    steps=len(test_generator))
# The index for prediction of testing set
patches_pred_labels = np.argmax(pred_prob, axis=1)
pred_labels = np.empty((NImages),dtype=np.int)
for idx_classes in range(0, NImages):
    idx_each = int(idx_classes * 49)
    # Obtain the all patches for NO.idx_classes image
    list_each = list(patches_pred_labels[idx_each:(idx_each+49)])
    # The aim is to rearrage the prediction order when the majority votes are the same
    # {'Benign': 0, 'In Situ': 1, 'Invasive': 2, 'Normal': 3}
    new_idx_classes = np.array([2, 1, 0, 3], dtype=np.int)
    counter_each_classes = np.array([list_each.count(2), list_each.count(1), list_each.count(0), list_each.count(3)])
    pred_labels[idx_classes] = new_idx_classes[np.argmax(counter_each_classes)]

# The true labels of testing set
patches_true_labels = test_generator.classes
true_labels = np.empty((NImages),dtype=np.int)
for idx_classes in range(0, NImages):
    idx_each = int(idx_classes * 49)
    true_labels[idx_classes] = patches_true_labels[idx_each]


# Print classification results
cfm = confusion_matrix(true_labels, pred_labels)
print(classification_report(true_labels, pred_labels, digits=4))
