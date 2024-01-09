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

Find parameters, w and b that minimizes the cost function, $J(w,b)$. There is an algorithm for doing this called gradient descent. This algorithm is one of the most important algorithms in machine learning. Gradient descent and variations on gradient descent are used to train, not just linear regression, but some of the biggest and most complex models in all of AI.

## Gradient Decent

_gradient descent_ was described as:

$$
\begin{align*} \text{repeat}&\text{ until convergence (where w and b does not change much):} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}
$$

where, parameters $w$, $b$ are updated _simultaneously_.  
The gradient is defined as:

$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$

Here _simultaniously_ means that you calculate the partial derivatives for all the parameters before updating any of the parameters. That is:

$$
temp_w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline
 temp_b &= b -  \alpha \frac{\partial J(w,b)}{\partial b} \; \newline

 w = temp_w \; \newline
 b = temp_b \newline \rbrace
\end{align*}
$$

$\alpha$ is the learning rate and is between 0 and 1 and controls the step taken in each iteration. If $\alpha$ is too small then the gradient decent will be too slow and if too large, it overshoots the minimum and may fail to converge. So if your parameters is already in a local minimum, then further gradient descent steps do absolutely nothing. The term batch gradient descent refers to the fact that on every step of gradient descent, we're using all of the training examples, instead of just a subset of the training data. An alternative way for finding w and b for linear regression is called the normal equation. This works only for linear regression (does not generalise to other learning algorithms) and does not use an iterative procedure and is generally slow when the number of features is large.

## Feature Scaling and Learning rate

when you have different features that take on very different ranges of values, it can cause gradient descent to run slowly but re-scaling the different features so they all take on comparable range of values. because this will speed up, gradient decent significantly. Feature scaling can be done via

1. normalizing with the maximum value
2. mean normalization $$x_1= (x_1 - \mu_1) \ (max - min)$$, $\mu_1$ is the mean of the feature 1.
3. z-score normalisation $$x_1= (x_1 - \mu_1) \ \sigma_1$$

## Checking gradient decent for convergence

plot learning curve (i.e. J(w,b) as a function of the number of iterations ). If gradient descent is working properly, then the cost J(w,b) should decrease after every single iteration. If J(w,b) ever increases after one iteration, that means either $\alpha$ is chosen poorly, and it usually means $\alpha$ is too large, or there could be a bug in the code.

## Feature Engineering

Using one's knowledge or intuition about the problem to design new features usually by transforming or combining the original features of the problem in order to make it easier for the learning algorithm to make accurate predictions.

## Polynomial Regression

Let's take the ideas of multiple linear regression and feature engineering to come up with a new algorithm called polynomial regression, which will let you fit curves, non-linear functions, to your data. By using feature engineering and polynomial functions, you can potentially get a much better model for your data.

## Classification with Logistic Regression

Logistic regression, which is probably the single most widely used classification algorithm in the world. The predictions of our classification model is between 0 and 1 since our output variable $y$ is either 0 or 1. This can be accomplished by using a "sigmoid function" which maps all input values to values between 0 and 1.

### Formula for Sigmoid function

The formula for a sigmoid function is as follows -

$$f(x) = g(z) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b ) = \frac{1}{1+e^{-z}}\tag{1}$$

In the case of logistic regression, z (the input to the sigmoid function), is the output of a linear regression model (i.e.\ $z = w\cdot x +b$ or more higher order polynonial function). The decision boundary is where $z = 0$. This boundary can be linear or non-linear

## Cost function for logistic regression

For linear regression the cost function looks like a convex function and non-convex function for a logistic regression function. For the **Linear** Regression we have used the **squared error cost function** (with one variable is):
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$

where
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$
For each input variable the cost function is actually called a loss function:

$$L(f_{w,b}(x^{(i)}, y^{(i)}))$$
**Loss** is a measure of the difference of a single example to its target value while the  
**Cost** is a measure of the losses over the training set

By choosing different forms of the loss function for our logistic regression, we can convert the non-convex function to a convex function. Naively the cost error function for the logistic regression is:
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

where
$$f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$$.

The new defined loss function so that the cost functon remains convex:

- $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:

\begin{equation}
loss(f*{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases} - \log\left(f*{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=1$}\\ - \log \left( 1 - f\_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=0$}
\end{cases}
\end{equation}

- $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value.

- $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot\mathbf{x}^{(i)}+b)$ where function $g$ is the sigmoid function.

The defining feature of this loss function is the fact that it uses two separate curves. One for the case when the target is zero or ($y=0$) and another for when the target is one ($y=1$). Combined, these curves provide the behavior useful for a loss function, namely, being zero when the prediction matches the target and rapidly increasing in value as the prediction differs from the target.

The loss function above can be rewritten to be easier to implement.
$$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$

The cost function for logistic regression is of the form

$$ J(\mathbf{w},b) = \frac{1}{m} \sum*{i=0}^{m-1} \left[ loss(f*{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}$$

where

- $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:

  $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$

- where m is the number of training examples in the data set and:
  $$
  \begin{align}
    f_{\mathbf{w},b}(\mathbf{x^{(i)}}) &= g(z^{(i)})\tag{3} \\
    z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\tag{4} \\
    g(z^{(i)}) &= \frac{1}{1+e^{-z^{(i)}}}\tag{5}
  \end{align}
  $$

## Gradient Descent for Logistic Regression

Recall the gradient descent algorithm utilizes the gradient calculation:

$$
\begin{align*}
&\text{repeat until convergence:} \; \lbrace \\
&  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1} \\
&  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}
$$

Where each iteration performs simultaneous updates on $w_j$ for all $j$, where

$$
\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3}
\end{align*}
$$

- m is the number of training examples in the data set
- $f_{\mathbf{w},b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target
- For a logistic regression model  
   $z = \mathbf{w} \cdot \mathbf{x} + b$  
   $f_{\mathbf{w},b}(x) = g(z)$  
   where $g(z)$ is the sigmoid function:  
   $g(z) = \frac{1}{1+e^{-z}}$

## Overfitting

If a model does not fit the training data well and has a high bias - called underfitting. If the model fits well to the data, we have generalization on new data set. In the extreme case where the model fits well on the training data set but does not generalise to new data set, hence it overfit and has high variance.

## Addressing overfitting

- collect more data.
- select features to include or exclude using one's intuition.
- regularization (control the impact of features without necessarily removing certain features).

## Cost functions with regularization

### Cost function for regularized linear regression

The equation for the cost function regularized linear regression is:
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2  + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{1}$$
where $\lambda$ is the regularization parameter and:
$$ f\_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b \tag{2} $$

Compare this to the cost function without regularization (which you implemented in a previous lab), which is of the form:

$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 $$

The difference is the regularization term, <span style="color:red">
$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$ </span>

Including this term encourages gradient descent to minimize the size of the parameters. Note, in this example, the parameter $b$ is not regularized. This is standard practice. if lambda is 0 this model will over fit If lambda is enormous like 10 to the power of 10. This model will under fit. Increasing the regularization parameter, lambda reduces overfitting by reducing the size of the parameters. For some parameters that are near zero, this reduces the effect of the associated features.

### Cost function for regularized logistic regression

For regularized **logistic** regression, the cost function is of the form
$$J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{3}$$
where:
$$ f\_{\mathbf{w},b}(\mathbf{x}^{(i)}) = sigmoid(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) \tag{4} $$

Compare this to the cost function without regularization (which you implemented in a previous lab):

$$ J(\mathbf{w},b) = \frac{1}{m}\sum*{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f*{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f\_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right] $$

As was the case in linear regression above, the difference is the regularization term, which is <span style="color:red">
$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$ </span>

# Neural Network (NN)

Neural network has a number of layers: input, hidden and output layers. The layers starting fro the first hidden layer to the output layer comprise of multiple units called neurons. The neuron make some computation on an input data and each returns what is called _activations_ (values). Unlike the tradition regression models where we do feature engineering, neural networks have the advantage that they learn their own features to make the learning problem easier. Another advantage of NN over traditional AI is an improve performance with increasing data data and network size. **Choosing the right number of hidden layers and number of hidden units per layer can have an impact on the performance of a learning algorithm as well.**
Steps in the NN architecture

1. model = Sequential([
   Dense (units=10, activation ="ReLU")
   Dense (units=15, activation ="ReLU")
   Dense (units=1, activation ="sigmoid")
   ])
2. model.compile(loss = BinaryCrossentropy()). For logistic regression in tensorflow we use $loss = BinaryCrossentropy()$ or $loss = MeanSquareError()$ for linear regression

3. model.fit(X,y epochs=100). This is minimizing the cost function to come up with the best fit parameters.

Computing the derivative for gradient decent is done using "backpropagation". Some example activation functions: linear activation function, sigmoid, rectified linear unit (ReLU) and softmax. You can choose different activation functions for different neurons in your neural network, and when considering the activation function for the output layer, it turns out that there'll often be one fairly natural choice, depending on what is the target or the ground truth label y. If you are working on a classification problem where y is either zero or one, so a binary classification problem, then the sigmoid activation function will almost always be the most natural choice. If y can be either positive or negative, and in that case you use the linear activation function. if y can only take on non-negative values, then the most natural choice will be the ReLU activation function. ReLU is the most common choice for hidden layers because it is faster to compute.

## Multi-Class (Softmax) Classification

the target, y can take on more than two possible values. Softmax is a generalisation of a logistic regression (binary classification) problem to a multi-class context. The softmax function can be written:
$$a_j = \frac{e^{z_j}}{ \sum_{k=1}^{N}{e^{z_k} }} \tag{1}$$

The loss function associated with Softmax, the cross-entropy loss, is:
\begin{equation}
L(\mathbf{a},y)=\begin{cases}
-log(a\*1), & \text{if $y=1$}.\\
&\vdots\\
-log(a_N), & \text{if $y=N$}
\end{cases} \tag{3}
\end{equation}
Where y is the target category for this example and $\mathbf{a}$ is the output of a softmax function. In particular, the values in $\mathbf{a}$ are probabilities that sum to one. To write the cost equation we need an 'indicator function' that will be 1 when the index matches the target and zero otherwise.

$$
\mathbf{1}\{y == n\} = =\begin{cases}
    1, & \text{if $y==n$}.\\
    0, & \text{otherwise}.
  \end{cases}
$$

Now the cost is:
\begin{align}
J(\mathbf{w},b) = -\frac{1}{m} \left[ \sum*{i=1}^{m} \sum*{j=1}^{N} 1\left\{y^{(i)} == j\right\} \log \frac{e^{z^{(i)}\_j}}{\sum*{k=1}^N e^{z^{(i)}\_k} }\right] \tag{4}
\end{align}

Where $m$ is the number of examples, $N$ is the number of outputs. This is the average of all the losses.

<code> 
model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
X_train,y_train,
epochs=10
)
</code>

A better implementation is:
<code>
model = Sequential(
[
Dense(25, activation = 'relu'),
Dense(15, activation = 'relu'),
Dense(4, activation = 'linear') # < softmax activation here
]
)
model.compile(
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
X_train,y_train,
epochs=10
)
</code>

The adam algorithm can adjust the learning rate automatically or opimise it.

## Layer types

1. Desnse layer: each neuron output is a function of all the activation outputs of the previous layer
2. Convolutional layer: each neuron only looks at part of the previous layer's output. Faster computation, needs less training data (less prone to overfitting)

## Evaluating a model

Divide data into training and test set and compute test and train error. Try different models and choose the model with the least test error. This procedure produces overly optimistic estimate of the eneralistion error on the trainin data. Another procedure is to split the data into training, cross validation and test set. Calculate the cross validation error and choose the model with the least cross-validation (validation, development, dev) error and finally estimate the generalistion error using the test set.

**module 2 week 3\***.
A code example for model evaluation is given:
<code>

# Initialize lists containing the lists, models, and scalers

train_mses = []
cv_mses = []
models = []
scalers = []

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.

for degree in range(1, 11):

    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)

    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)

    # Add polynomial features and scale the cross validation set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)

</code>
## What is deep learning?

So what is deep learning? Deep learning uses an algorithm called neural networks, which are loosely inspired by biological neural networks in human brains. Neurons, also sometimes called nodes, are the basic unit of neural networks. Deep learning is a special type of machine learning that can solve more complex problems but it requires much more data than traditional machine learning. It is best used in cases where the inputs are less structured, such as large amounts of text or images.

## Computer vision

So, what is computer vision? The goal of computer vision is to help computers see and understand the content of digital images. Computer vision is necessary to enable, for example, self-driving cars. Manufacturers such as Tesla, BMW, Volvo, and Audi use multiple cameras to acquire images from the environment so that their self-driving cars can detect objects, lane markings, and traffic signs to safely drive.

eg: Face recognition

Just like before, the neurons in the middle will compute various values by themselves. Typically, when feeding a neural network images, neurons in the earlier stages will learn to detect edges later on parts of objects, like eyes and noses for example, the final neurons will learn to detect shapes of faces. In the end, the network will put all of this together to output the identity of the person in the image.
