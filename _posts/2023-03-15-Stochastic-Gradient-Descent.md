---
layout: single
classes: wide
title: "Gradient Descent"
author: Erick Platero
category: DL
tags: [dl]
author_profile: true

---
# Gradient Descent

### Objective

In most cases, the objective in ML and DL boil down to one thing: **minimizing error**.

This error is computed by feeding our model some data, recording its predictions, and feeding it to a criterion, which evaluates the performance or error of the model so that we can backpropagate/train it. We will call the operation that evaluates performance our Loss function.

<img src="https://github.com/Erick7451/DL-with-PyTorch/blob/master/img/image-20201022230530198.png?raw=true" alt="image-20201022230530198.png" style="zoom:80%;" />

### Gradient Descent

There are a multitude of strategies to minimize the Loss function, all with their own unique properties; however, in DL, the objective is often approximated by some variation of Gradient Descent.

**Why?**

In essence, Gradient Descent is a small "step" towards the path with steepest descent (hence its name) in your model's function. 

There are three main ingredients needed to take this step:

1. A differentiable function/model ($\frac{\partial F}{\partial w_i}$)
2. Partial of Loss function w.r.t. model parameters ($\frac{\partial L}{\partial w_i}$)
3. A learning rate ($\alpha$)

Once instantiated, we formulate Gradient Descent as:

$$
w_i = w_i - \frac{\partial L}{\partial w_i} * \alpha \tag{1}
$$

More succinctly: 

![image-20201023080311021.png](https://github.com/Erick7451/DL-with-PyTorch/blob/master/img/image-20201023080311021.png?raw=true)

> **NOTE:** Without a differentiable function ($\frac{\partial F}{\partial w_i}$), we would not be able to take Partial of Loss function w.r.t. model parameters ($\frac{\partial L}{\partial w_i}$)

It's a simple formulation but a powerful technique. 

Now, Gradient Descent is often referred to as "stochastic" in reference to the lack of RAM storage that is needed to hold the usually large amounts of data needed to train Deep Learning models. As such, we are instead forced to create batches of (usually) randomly distributed data. When we do this, given that we are training our model on different distributions of data, our model will look like we are taking random or stochastic steps that will slowly converge to a minimum. 

I mentioned that this technique "slowly" converges because gradient descent is only effective when it is constantly iterated through thousands of cycles due to the multiplication with a small $\alpha$, which forces our "step" to be tiny (I will expand more on this in the next section).

So, **why does it work?**

### Theory

Let's revisit the definition of the derivative for a bit

$$
\frac{dy}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h} \tag{2}
$$

From x, if we take a small forward step $h$, how much will $y$ increase/decrease by?

Traditionally, this formulation is interpreted as the velocity of the function at $x$. However, we can reformulate our expression through some algebra so lthat it can actually be interpreted as a linear approximation of $f(x+h)$ from our initial value $(x, f(x))$ to an arbitrary $x+h$. 

$$
\begin{align*}
& \frac{dy}{dx} = \lim_{h\to0}\frac{f(x+h)-f(x)}{h} \tag{2} \\
& \frac{dy}{dx}*\lim_{h\to0}h = f(x+h)-f(x) \\
& f(x+h) - f(x) = \frac{dy}{dx}*\lim_{h\to0}h  \\
& f(x+h) = f(x) + \frac{dy}{dx}*\lim_{h\to0}h  \\
& f(x+h) \approx f(x) + \frac{dy}{dx} * h \tag{3}
& \end{align*}
$$

> **NOTE:** the limit was taken off so as to give freedom to approximate $f(x+h)$ at any given step $h$

Does this formulation look familiar? It should! It looks ***very similar*** to our "step" function in gradient descent! 
$$
\begin{align}
w_i &= w_i - \frac{\partial L}{\partial w_i} * \alpha \ (\text{Gradient Descent}) \tag{2}\\
f(x+h) &\approx f(x) + \frac{dy}{dx} * h \ (\text{Linear Approximation}) \tag{3}
\end{align}
$$

That is because we are applying an equal formulation to a different problem.

For Gradient Descent, instead of taking a step forward as we do with our Linear Approximation, we are actually taking a step backward (hence, the negative sign). 

As a result, in Gradient Descent we are approximating the step towards steepest descend, while in our Linear Approximation, we are approximating the step towards steepest ascent.

> **TIP**: To understand why the gradient is the direction of steepest ascent/descent, please refer to [this](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/why-the-gradient-is-the-direction-of-steepest-ascent#:~:text=As%20a%20consequence%2C%20it%27s%20the%20direction%20of%20steepest,that%20you%20could%20want%20a%20derivative%20to%20extend.) great explanation offered by Khan Academy 

### Criterion

Now that we are able to attain our linear approximation of the steepest descent or ascent of our function, we must next: 


* decide what Loss function we will use to compare the truth from our prediction and
* understand the relationship between our Loss function and alpha $\alpha$ 

Let us treat our problem as one of regression and use the Squared Error (SE) to compute the Loss of our prediction based on the truth:
$$
\begin{align}
\text{Squared Error} &= (f(x) + \frac{dy}{dx}h-f(x+h))^2 \ (\text{Steepest Ascent}) \tag{4} \\
\text{Squared Error} &= (f(x) - \frac{dy}{dx}h-f(x+h))^2 \ (\text{Steepest Descent}) \tag{5}
\end{align}
$$

The definition of the derivative we saw earlier assumes the following:

$h\to 0$, $\text{Error}\to0$

Conversely, if we are approximating a non-linear function, below is generally true:

$h\to\infty$, $Error\to\infty$

Let us display this concept with a simple example of approximating any step of $f(x+h)$ from $x=4$ by modeling $f(x)=x^2$


<iframe src="/assets/html/fig1.html" width="100%" height="500px"></iframe>

> **NOTE**: All visualizations can be found at the end of this notebook

The further of a step $h$ we make from $x=4$, the higher is our Squared Error! Conversely, however, when we take a small step, our squared error is minimized.

Now, keep in mind that from the perspective of Gradient Descent, we (usually) do ***not calculate the Error of our linear approximation*** as shown above.

Personally, this is due to the following:

* Our objective is to minimize the error w.r.t. our model's predictions from the truth, not calculate the error associated with our Gradient Descent "step"
* Calculating the 

Hence, despite just taking a small step "as is" in gradient descent, we want to minimize our error so as to accurately converge to our local minima.

### The Trade-Off

Given that a small step $h$ minimizes our modeling error, it follows to place a small learning $\alpha$ while performing Gradient Descent. However, a very small $\alpha$ may take hundreds or thousands of iterations more to reach a local minima. This is very problematic when our model is a "deep" network that takes much computation time to compute (as most up-to-date models are)

Hence, when setting $\alpha$ for our learning rate, we must have a trade-off between our

1. Tolerance of error and 
2. The time it takes to reach anything close to our local minima

Let's view below animations to get a better understanding of these concepts


<iframe src="/assets/html/fig2.html" width="100%" height="500px"></iframe>

First, notice that as the graphs progress, the learning rate of .1 converges **significantly** faster than both .01 and .001, despite having a higher level of tolerance for error. 

Second, as learning rates begin to reach the global minima, convergence begins to plateau as the rate $dy/dx$ begins to decrease (thereby leading to smaller changes)

Third, while it may seem intuitive to place a high learning rate to our "well-behaved" quadratic function, in practice, the higher level of tolerance may lead us to **"overshoot"** our local minima. 

<img src="https://yogayu.github.io/DeepLearningCourse/04/img/gd4.png" alt="Learning to minimize error—Gradient Descent Method" style="zoom: 20%;" />

Hence, when inserting $\alpha$, we must find that "golden mean" where our progression is not too fast nor too slow. There is no methodological way to find this other than pure trial and error. However, careful thought should go on in placing learning rates as computer power may restrict us to efficiently experiment between two or three learning rates. 

### Experiment

Now that we have defined Gradient Descent, we will showcase its performance on booth's function, which is categorized as a Plate function.

We will:

1. Create a class object that holds the forward and backward pass of the function,
2. Perform gradient descent at $\alpha=.01$ for 150 epochs and
3. Graph our results

> **NOTE**: The code that performed the above steps is found at the end of this notebook.

Now, let us visualize our results








From the above visualizations, we can tell that Gradient Descent worked well! At 150 epochs, we reached the local minima of our plate function. 

### Newton's Method

Again, Gradient Descent is a very simple function that relies on heavy iterations to reach a local minima. However, are there other methods? 

Yes, Newton's method is one of them.

Newton's method uses the equivalent formulation of Gradient Descent, however, it applies it differently.

In a nutshell, Newton's method is the same derivation of Gradient Descent, however, this time applied to the **derivative of the function** such as shown below:

$$
\nabla f(x+h) \approx \nabla f(x) + \nabla^2f(x)h
$$

Now, what makes Newton's method truly unique is that it approximates the **step** $h$ needed to ***reach the root of our gradient function*** such as shown below:

$$
0 = \nabla f(x+h) \approx \nabla f(x) + \nabla^2f(x)h
$$

Then, solving for **step** $h$ we get:

$$
h=-(\nabla^2f(x))^{-1}\nabla f(x)
$$

Let us shown an animation that represents a "step" in Newton's method


<iframe src="/assets/html/fig3.html" width="100%" height="500px"></iframe>


Already, just one-step of Newton's Method greatly approximates the global minima. 

From this perspective, Newton's method is no doubt superior than gradient descent. However, how come it is not widely applied in DL?

Newton's method forces us to compute the Hessian matrix, which is very expensive. The cost of computation coupled with some divergence problems has refrained most users from this approach. 

Although there have been remedies of approximating the Hessian matrix for a boost in calculation speed, it remains to be a difficult to control optimization algorithm. 

# Conclusion

Optimization algorithms represent the "training" phase of DL. It is crucial to understand such an important concept, especially when it becomes a "very easy" function to implement on frameworks like PyTorch or TensorFlow.

An understanding will expand the realm of new possible training algorithms that are waiting to be found and will ensure the user to make "well-informed" decisions of our learning rate parameter.

That's it for this tutorial! Below this cells is a "Graph" section where one can find all the codes that were used for graphing.

# Where to Next?

**Activation Functions Tutorial:**
https://nbviewer.jupyter.org/github/Erick7451/DL-with-PyTorch/blob/master/Jupyter_Notebooks/ReLU.ipynb
