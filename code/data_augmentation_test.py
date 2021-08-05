import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from SimCLR.data_util import *


from tensorflow.keras import layers

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
image_cp = np.copy(image)
plt.imshow(image)
plt.title(get_label_name(label))
#plt.imsave("../media/tulips.pdf",image)
plt.show()

# Add the image to a batch for data augmentation
image = tf.expand_dims(image, 0)

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

augmented = data_augmentation(image)

plt.imshow(augmented[0])
#plt.imsave("../media/tulips-rot1.pdf",augmented[0])
plt.show()


new = np.array(np.copy(image_cp),dtype =np.uint8 )
new = random_color_jitter(new,strength = 0.25)
plt.imshow(new)
plt.imsave("../media/tulips-colorjitter.pdf",new)
plt.show()


import random
import cv2

def random_crop_and_resize(image, size=224):

    image = resize_image(image)

    h, w = image.shape[:2]

    y = np.random.randint(0, h-size)
    x = np.random.randint(0, w-size)

    image = image[y:y+size, x:x+size, :]

    return image


def resize_image(image, size=224, bias=5):

    image_shape = image.shape

    size_min = np.min(image_shape[:2])
    size_max = np.max(image_shape[:2])

    min_size = size + np.random.randint(1, bias)

    scale = float(min_size) / float(size_min)

    image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

    return image


orig = np.copy(image_cp)
# Second augmentation
augmented_2 = random_crop_and_resize(image_cp)
augmented_2 = cv2.resize(augmented_2,(orig.shape[1],orig.shape[0]))

plt.imshow(augmented_2)
#plt.imsave("../media/tulips-randomcrop.pdf",augmented_2)
plt.show()


# Histograms
histr1 = cv2.calcHist([orig],[0],None,[256],[0,256])
histr2 = cv2.calcHist([augmented_2],[0],None,[256],[0,256])

# show the plotting graph of an image
plt.plot(histr1,label="original")
plt.plot(histr2,label="random-crop")
plt.legend(loc="upper left")
#plt.savefig("../media/tulips-histogram.pdf")
plt.show()

# Histograms of cropped vs jittered cropped
histr3 = cv2.calcHist([np.array(new)],[0],None,[256],[0,256])
plt.plot(histr2,label="cropped")
plt.plot(histr3,label="color-jitter")
plt.legend(loc="upper right")
#plt.savefig("../media/histogram-croppedvsjitter.pdf")
plt.show()
