import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## load_data cached in ~/.keras/datasets/fashion-mnist
# import pickle
# 
# with open('data.pkl', 'wb') as f:
#     pickle.dump(((train_images, train_labels), (test_images, test_labels), ), f)
