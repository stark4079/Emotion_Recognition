
# coding: utf-8

# In[8]:


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.models import load_model
import keras.backend as K
import argparse
import os

from AMOL.nn.conv.emotionvggnet import EmotionVGGNet


# import the necessary packages

from config import emotion_config as config
from AMOL.hdf5datasetgenerator import HDF5DatasetGenerator
from AMOL.imagetoarraypreprocessor import ImageToArrayPreprocessor
from AMOL.callbacks.epochcheckpoint import EpochCheckpoint
from AMOL.callbacks.trainingmonitor import TrainingMonitor


# In[ ]:


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
        help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
        help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
        help="epoch to restart training at")
ap.add_argument("-lr", "--learning-rate", type=float, default=1e-2,
        help="learning rate")
ap.add_argument("-opt", "--optimizer", type=str, default="Adam",
        help="optimizer of model")
ap.add_argument("-e", "--epoch", type=int, default=20,
        help="number of epochs")
args = vars(ap.parse_args())


# In[ ]:


# construct the training and testing image generators for data
# augmentation, then initialize the image preprocessor
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
        horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()


# In[ ]:


# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
        aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
        aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)


# In[ ]:


# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1,
            classes=config.NUM_CLASSES)
    opt = Adam(learning_rate=args["learning_rate"])
    if args["optimizer"] =="SGD":
        opt = SGD(learning_rate=args["learning_rate"] ,momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])
# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, args["learning_rate"])
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))


# In[ ]:


# construct the set of callbacks
figPath = os.path.sep.join([config.OUTPUT_PATH,
    "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH,
    "vggnet_emotion.json"])
callbacks = [
        EpochCheckpoint(args["checkpoints"], every=5,
            startAt=args["start_epoch"]),
        TrainingMonitor(figPath, jsonPath=jsonPath,
            startAt=args["start_epoch"])]


# In[ ]:


# train the network
model.fit(trainGen.generator(),
        steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // config.BATCH_SIZE,
        epochs=args["epoch"],
        max_queue_size=config.BATCH_SIZE * 2,
        callbacks=callbacks, verbose=1)

# close the databases
trainGen.close()
valGen.close()

