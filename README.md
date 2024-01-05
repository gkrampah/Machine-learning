# Notes on Machine learning

Machine learning is the science and art of giving computers the ability to learn to make decisions from data without being explicitly programmed (Arthur Samuel 1959). For example, your computer can learn to predict whether an email is spam or not spam given its content and sender. Another example: your computer can learn to cluster, say, Wikipedia entries, into different categories based on the words they contain. Notice that, in the first example, we are trying to predict a particular class label, that is, spam or not spam. In the second example, there is no such label. When there are labels present, we call it supervised learning. When there are no labels present, we call it unsupervised learning. There are two main commonly used types of machine learning: supervised (used in most real-world applications and has the most rapid advancement) and unsupervised learning. Reinforcement learning is the third and less popularly used ML algorithm.

Unsupervised learning, in essence, is the machine learning task of uncovering hidden patterns and structures from _unlabeled data_. That is the data comes with only inputs, x and not output labels, y. For example, a business may wish to group its customers into distinct categories based on their purchasing behavior without knowing in advance what these categories maybe. This is known as clustering, one branch of unsupervised learning. Another example is anomaly detection (find unusual data points) and dimensionality reduction (take a big data-set and almost magically compress it to a much smaller data-set while losing as little information as possible.)

Supervised learning algorithms learn the input to output mappings using examples of labelled data pairs (feature, x and target y) and predicts y for a certain new input x. In supervised learning, we have several data points or samples, described using predictor variables or features and a target variable. Our data is commonly represented in a table structure such as the one you see here, in which there is a row for each data point and a column for each feature. The aim of supervised learning is to build a model that is able to predict the target variable.

If the target variable consists of categories, like 'click' or 'no click', 'spam' or 'not spam', or different species of flowers, we call the learning task classification. Alternatively, if the target is a continuously varying variable, for example, the price of a house, it is a regression task. The classification algorithms predict categories or a small limit set of possible outputs while regression task predict a number and has an infinite number of possible outputs. The training input can consist of two or more features

There are three types of machine learning. The first listed, reinforcement learning, is used for deciding sequential actions, like a robot deciding its path or its next move in a chess game. Reinforcement learning is not as common as the others and uses complex mathematics, like game theory. The most common types are supervised and unsupervised learning. Their main difference lies in their training data.

In supervised learning, the training data is "labeled", meaning the values of our target are known. For instance, we knew if previous patients had heart disease based on the labels "true" and "false". In unsupervised learning, we don't have labels, only features. What can we do with this? Usually tasks like anomaly detection and clustering, which divides data into groups based on similarity. Let's explore this with our dataset.

There are two flavors of supervised learning: classification and regression. Classification consists in assigning a category to an observation. We're predicting a discrete variable, a variable that can only take a few different values. Is this customer going to stop its subscription or not? Is this mole cancerous or not? Is this wine red, white or rosé? Is this flower a rose, a tulip, a carnation, a lily? While classification assigns a category, regression assigns a continuous variable, that is, a variable that can take any value. For example, how much will this stock be worth? What is this exoplanet's mass? How tall will this child be as an adult?

Unsupervised learning is quite similar to supervised learning, except it doesn't have a target column - hence the unsupervised part. So what's the point then? Unsupervised learning learns from the dataset, and tries to find patterns. That's the reason this technique is so interesting and powerful: we can find insights without knowing much about our dataset.

## How training of supervised learning algorithm works

To train the model, feed the training dataset, both the input features and the output targets to your learning algorithm. Then your supervised learning algorithm will produce some function, f which take a new input x and outputs an estimate or a prediction, $y-hat$. In machine learning, the convention is that y-hat is the estimate or the prediction for y. The function f is called the model. X is called the input or the input feature, and the output of the model is the prediction, y-hat (and y is the true target). The model's prediction is the estimated value of y.

What is the math formula we're going to use to compute f?

The first key step is to define a cost function and the cost function will tell us how well the model is doing so that we can try to get it to do better. Note that, in machine learning parameters of the model are the variables you can adjust during training in order to improve the model (eg: $f_{w,b}= wx+b$, w and b are the parameters).

## Computing Cost

Here, cost is a measure how well our model is predicting the target. The equation for cost with one variable is:
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$

where
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$

- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.
- $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ is the squared difference between the target value and the prediction.
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.
  > Note, in lecture summation ranges are typically from 1 to m, while code will be from 0 to m-1.

Choose parameters, w and b that minimizes $J(w,b)$

What is deep learning?

So what is deep learning? Deep learning uses an algorithm called neural networks, which are loosely inspired by biological neural networks in human brains. Neurons, also sometimes called nodes, are the basic unit of neural networks. Deep learning is a special type of machine learning that can solve more complex problems but it requires much more data than traditional machine learning. It is best used in cases where the inputs are less structured, such as large amounts of text or images.

Computer vision

So, what is computer vision? The goal of computer vision is to help computers see and understand the content of digital images. Computer vision is necessary to enable, for example, self-driving cars. Manufacturers such as Tesla, BMW, Volvo, and Audi use multiple cameras to acquire images from the environment so that their self-driving cars can detect objects, lane markings, and traffic signs to safely drive.

eg: Face recognition

Just like before, the neurons in the middle will compute various values by themselves. Typically, when feeding a neural network images, neurons in the earlier stages will learn to detect edges later on parts of objects, like eyes and noses for example, the final neurons will learn to detect shapes of faces. In the end, the network will put all of this together to output the identity of the person in the image.
