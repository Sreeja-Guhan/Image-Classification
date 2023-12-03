import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.ERROR)
#print('Using TensorFlow version', tf.__version__)


### Import MNIST


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""### Shapes of Imported Arrays"""

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""### Plot an Image Example"""

# Commented out IPython magic to ensure Python compatibility.
#To check the image
from matplotlib import pyplot as plt
# %matplotlib inline

plt.imshow(x_train[0], cmap='binary') #binary because the shades are b&w
plt.show()
plt.imshow(x_train[6]) #without cmap
plt.show()

"""### Display Labels"""

y_train[0], y_train[6]

#View all the unique lables
set(y_train)

"""# Task : One Hot Encoding
After this encoding, every label will be converted to a list with 10 elements and the element at index to the corresponding class will be set to 1, rest will be set to 0:

| original label | one-hot encoded label |
|------|------|
| 5 | [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] |
| 7 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] |
| 1 | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] |

### Encoding Labels
"""

#This can be achieved using to_catgorical from TensorFlow
#Otherwise, use a for loop to manual encode
from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

"""### Validated Shapes"""

y_train_encoded.shape, y_test_encoded.shape

"""### Display Encoded Labels"""

y_train_encoded[0], y_train_encoded[6]

"""# Task : Neural Networks

### Linear Equations

![Single Neuron](images/1_2.png)

The above graph simply represents the equation:

\begin{equation}
y = w1 * x1 + w2 * x2 + w3 * x3 + b
\end{equation}

Where the `w1, w2, w3` are called the weights and `b` is an intercept term called bias. The equation can also be *vectorised* like this:

\begin{equation}
y = W . X + b
\end{equation}

Where `X = [x1, x2, x3]` and `W = [w1, w2, w3].T`. The .T means *transpose*. This is because we want the dot product to give us the result we want i.e. `w1 * x1 + w2 * x2 + w3 * x3`. This gives us the vectorised version of our linear equation.

A simple, linear approach to solving hand-written image classification problem - could it work?

![Single Neuron with 784 features](images/1_3.png)

### Neural Networks

![Neural Network with 2 hidden layers](images/1_4.png)

This model is much more likely to solve the problem as it can learn more complex function mapping for the inputs and outputs in our dataset.

# Task : Preprocessing the Examples

### Unrolling N-dimensional Arrays to Vectors
"""

import numpy as np

x_train_reshaped = np.reshape(x_train, (60000,784))
x_test_reshaped = np.reshape(x_test, (10000,784))
x_train_reshaped.shape, x_test_reshaped.shape

"""### Display Pixel Values"""

print(set(x_train_reshaped[0]))

"""### Data Normalization"""

x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_test_reshaped)
epsilon = 1e-10 #Why? Because if std dev is very small, it may lead to irregularities
#Hence, adding a small value epsilon can avoid that

x_train_norm = (x_train_reshaped-x_mean)/(x_std+epsilon)
x_test_norm = (x_test_reshaped-x_mean)/(x_std+epsilon)

"""### Display Normalized Pixel Values"""

print(set(x_train_norm[0]))

"""# Task : Creating a Model
### Creating the Model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

"""### Activation Functions

The first step in the node is the linear sum of the inputs:
\begin{equation}
Z = W . X + b
\end{equation}

The second step in the node is the activation function output:

\begin{equation}
A = f(Z)
\end{equation}

Graphical representation of a node where the two operations are performed:

![ReLU](images/1_5.png)

### Compiling the Model
"""

#Relu - Linear for all +ve values and set to 0 otherwise
#Softmax - Gives probability scores for each class
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',#subtraction btw true with predicted
    metrics=['accuracy']
)
model.summary()

"""# Task : Training the Model

### Training the Model
"""

model.fit(x_train_norm,y_train_encoded,epochs=4)

"""### Evaluating the Model"""

loss, accuracy = model.evaluate(x_test_norm,y_test_encoded)
print("Test accuracy:",accuracy*100)

"""# Task : Predictions

### Predictions on Test Set
"""

preds = model.predict(x_test_norm)
preds.shape

"""### Plotting the Results"""

plt.figure(figsize=(12,12))
start_index = 0
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  pred = np.argmax(preds[start_index+i])
  truth = y_test[start_index+i]
  color = 'g'
  if pred != truth:
    color = 'r'
  plt.xlabel(f"i={i},truth={truth},pred={pred}",color=color)
  plt.imshow(x_test[start_index+i])
plt.show()

plt.plot(preds[8])
plt.show()
#Softmax prob output for i=8

