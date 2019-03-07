#Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
#This is a computer vision example! Recognizing different items of clothing, trained from a dataset containing 10 different types!
# The Fashion MNIST data is available directly in the tf.keras dataset API.


import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist
#calling load_data on this object will give you two sets of two lists, these will be the training and testing vlaues for the graphics that contain the clothing items and their labels.
(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()
#All of the values in the number are between 0 and 255. If we are training a neural network, for vaious resons it's easier if we treat all values as between 0 and 1 (normalizing)
training_images = training_images/255.0
test_images = test_images/255.0
#design the model! sequential defines a sequence of layers in the neural network. Images are square and Flatten just takes that square and turns it into a 1 dimesional set. Dense adds a layer of neurons
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
