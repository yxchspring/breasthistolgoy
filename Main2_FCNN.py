from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import os
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import math
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import optimizers
from keras import regularizers
import pickle

######################### Step1:  Initialize the parameters and file paths #########################
# configure the parameters
batch_size = 20
num_classes = 4
epochs = 50
image_height = 384
image_width = 512
model_folder = 'FCNN'
# Configure the train, val, and test p
base_dir = 'D:/Data/Image/Biomedicine/integrated/breasthistology/Reinhard_Patches_CNN'
train_dir = os.path.join(base_dir, 'Training_data')
validation_dir = os.path.join(base_dir, 'Validation_data')
test_dir = os.path.join(base_dir, 'Test_data/Initial')
# test_dir = os.path.join(base_dir, 'Test_data/Extended')
# test_dir = os.path.join(base_dir, 'Test_data/Overall')

######################### Step2:  Construct the model #########################
# Construct the model
conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(image_height, image_width, 3))
conv_base.summary()
# The index and its layername
# [0, 'input_1']
# [1, 'block1_conv1']
# [2, 'block1_conv2']
# [3, 'block1_pool']
# [4, 'block2_conv1']
# [5, 'block2_conv2']
# [6, 'block2_pool']
# [7, 'block3_conv1']
# [8, 'block3_conv2']
# [9, 'block3_conv3']
# [10, 'block3_conv4']
# [11, 'block3_pool']
# [12, 'block4_conv1']
# [13, 'block4_conv2']
# [14, 'block4_conv3']
# [15, 'block4_conv4']
# [16, 'block4_pool']
# [17, 'block5_conv1']
# [18, 'block5_conv2']
# [19, 'block5_conv3']
# [20, 'block5_conv4']
# [21, 'block5_pool']
layer_conv_base = conv_base.layers

for layers_i in range(len(layer_conv_base)):
    print([layers_i,layer_conv_base[layers_i].name])

# configure the input layer
block5_pool_input = layer_conv_base[21].output
# block4_pool_input = layer_conv_base[16].output
block3_pool_input = layer_conv_base[11].output
block2_pool_input = layer_conv_base[6].output
# block1_pool_input = layer_conv_base[3].output

# add the batch_normalization layer
# block5_pool_input = layer_conv_base[18].output
# layers.
# For block5_pool
block5_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block5_pool_input)

# For block3_pool
block3_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block3_pool_input)
block3_pool_norm = layers.AveragePooling2D(pool_size=(4, 4))(block3_pool_norm)

# For block1_pool
block2_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block2_pool_input)
block2_pool_norm = layers.AveragePooling2D(pool_size=(8, 8))(block2_pool_norm)

# For block1_pool
# block1_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block1_pool_input)
# block1_pool_norm = layers.AveragePooling2D(pool_size=(16, 16))(block1_pool_norm)

output_fusion_norm = layers.concatenate(
    [block5_pool_norm,block3_pool_norm,block2_pool_norm],
    axis=-1)

# output_fusion_conc = layers.Lambda(expand_dim_backend)(output_fusion_norm)
output_fusion_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(output_fusion_norm)

# Configure the fully-connected layers
FC_output = layers.GlobalAveragePooling2D()(output_fusion_norm)
FC_output = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001))(FC_output)
FC_output = layers.Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.001))(FC_output)
FC_output = layers.Dense(num_classes, activation='sigmoid')(FC_output)

model = Model(inputs=conv_base.input, outputs=FC_output)
model.summary()

for layers_i in range(model.layers.__len__()):
    print([layers_i,model.layers[layers_i].name])

# freeze the conv_base
# conv_base.trainable = False

# Freeze the 0-18 layers
for layer in model.layers[0:18]:
    layer.trainable = False

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              # optimizer=opt,
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# Data preprocessing
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150*150
        target_size=(image_height, image_width),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_height, image_width),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')

# print the data and labels batch shape
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
filepath_ckp = os.path.join(checkpoint_dir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

# save the best model currently
checkpoint = ModelCheckpoint(
    filepath_ckp,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max')

# fit setup
print('The traning starts!\n')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples/batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=math.ceil(validation_generator.samples/batch_size),
    callbacks=[checkpoint],
    verbose=0)



######################### Step3:  Save the history data and plots #########################
# plot the acc and loss figure and save the results
plt_dir = os.path.join(os.getcwd(), model_folder, 'plots')
if not os.path.isdir(plt_dir):
    os.makedirs(plt_dir)

print('The ploting starts!\n')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# save the history
history_dir = os.path.join(os.getcwd(), model_folder, 'history')
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)

# wb: Write in binary
data_output = open(os.path.join(history_dir,'history_B235.pkl'),'wb')
pickle.dump(history.history,data_output)
data_output.close()

# rb: Read in binary
data_input = open(os.path.join(history_dir,'history_B235.pkl'),'rb')
read_data = pickle.load(data_input)
data_input.close()

epochs_range = range(len(acc))
plt.plot(epochs_range, acc, 'ro', label='Training acc')
plt.plot(epochs_range, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'acc.jpg'))
plt.figure()

plt.plot(epochs_range, loss, 'ro', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'loss.jpg'))
plt.show()
