#build a neural network that recognizes handwriting digits!
#Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train=x_train/255
x_test=x_test/255

plt.imshow(x_train[0])
print(y_train[0])
print(x_train[0])
model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),tf.keras.layers.Dense(512, activation=tf.nn.relu),tf.keras.layers.Dense(10, activation=tf.nn.softmax)   
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20,callbacks=[callbacks])
model.evaluate(x_test, y_test)

