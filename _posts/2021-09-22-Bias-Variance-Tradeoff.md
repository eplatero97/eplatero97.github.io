---
layout: single
title: "Bias-Variance Tradeoff"
author: Erick Platero
category: stats
tags: [stats]
author_profile: true 
---

# Bias-Variance Tradeoff

Before I begin, let us decompose $MSE$ into its bias/variance form. 
$$
\begin{equation} \label{equation1}
\begin{split}
MSE & = L(f(X), y) = L(\hat{y}, y)\\
& = E[(\hat{y} - y)^2] \\
& = E[(y-E[\hat{y}] +E[y] - \hat{y})^2] \\
& = E[(y-E[\hat{y}])^2] + E[(E[\hat{y}]-\hat{y})^2] - 2E[(y-E[\hat{y}])(E[\hat{y}]-\hat{y})] \\
& = E[(y-E[\hat{y}])^2] + E[(E[\hat{y}]-\hat{y})^2] \\
& = bias^2 + varince
\end{split}
\end{equation}
$$

**Bias-Variance Tradeoff?**

The bias-variance tradeoff has to do with the fact that we can represent MSE as a decomposition between bias and variance. Thus, to minimize MSE is equivalent to minimizing both bias and variance. However, bias and variance are difficult components to minimize simultaneously as there are three strict criteria that our data and model must both satisfy:

1. Variance of model predictions must be minimal (minimizes variance)
   * $min(var_{\hat{y}}) \implies min(E[\hat{y}] - \hat{y}) $
   * The more clustered our predictions are, the better estimate $E[\hat{y}]$ will be.
2. Variance of ground-truth must also be minimal (minimizes bias)
   * $min(var_{y}) \implies min(E[y] - y) $
   * The more clustered our data is, the better estimate $E[\hat{y}]$ will be if we assume below condition
3. Expected value of model predictions and ground-truth data are roughly equivalent (minimizes bias)
   * $E[\hat{y}] \approx E[y]$
   * The more similar the expected values between predictions and ground-truth, the less our bias will be

If all above conditions are met, then this implies that the distribution of our ground-truth data is very similar to the distribution of our predictions. Further, this distribution also implies that model predictions and ground-truth data are all clustered tightly in some area in space. 

Given that these conditions assume qualities inherent to the data (which is uncontrollable) and the model (which depends on the learning algorithm), satisfying all criteria will be unlikely. As such, for any condition we cannot satisfy, a tradeoff between bias and variance may occur. Let us enumerate the different scenarios that may arise when minimizing MSE:

* High variance, high bias (worst-case scenario)
  * None of the three conditions are met.
* Low variance, high bias (average-case scenario)
  * 1st condition is met, but no others.
* Low bias, high variance (average-case scenario)
  * 2nd and 3rd conditions are met, but not the 1st.
* Low variance, low bias (best-case scenario)
  * All three conditions are met.

From the above, we can see that having a low variance does not necessarily imply a high bias (and vice-versa). If one low quality does not necessarily imply a high quality on its counter-part, then why is there a so called "bias-variance tradeoff" ?

For starters, if the variance of the data is inherently high, then we would expect our learning algorithm to mimic a similar (and hopefully accurate) high variance distribution. This would imply that we would have to forego meeting the first two conditions while *possibly* satisfying the third. In such scenario then, variance is increased while bias is *possibly* decreased. However, keep in mind that if our model is able to mimic the high variance of the data, then the model will *most likely* have a low bias as being able to mimic high variance often implies that the model is able to approximate the expected value of the ground-truth data very well                   ($E[\hat{y}] \approx E[y]$).

Alternatively, if the variance distribution of our data is low and our algorithm mimics such distribution, then we will be able to satisfy the first and second condition while *possibly* satisfying the third. We can intuitively think of this scenario as putting all our "eggs in one basket" as the further the expected predictions are from the expected data (the more we violate the third condition), the higher the bias will be. For example, assume we have a dataset with $n$ number of samples and we assume our predictions and ground-truth data are compact (low-variance). Then,  assume $E[\hat{y}] = 40$ and $E[y] = 50$. Since our data is compact, most prediction values equal 40 and most ground-truth values equal 50. Thus, $error \approx (\hat{y} - y)n$. Most of this error is due to high bias. Consequently, the higher the distance of the expected values with low-variance data, the error will linearly increase by a constant $n$ . However, notice that since our data is tightly compacted, we could just use the expected value of the ground-truth to attain an error close to zero ($\sum{E[y] - y} \approx 0$). This generally showcases that simpler models (e.g., linear regression) may be less-apt to correctly model ground-truth distributions than more complex models (e.g., random forests) but are also able to minimize variance. 

Both of the above scenarios reveal the following general relationships between model complexity and data qualities:

* The higher the variance of the ground-truth data, the more complex our model will have to be to mimic the spread (which results in a low bias most likely due to the model not making many assumptions about the data).
  * E.g., random forests take a greedy approach (does not make much of any other assumptions).
* The higher the bias, the simpler our model will be (which results in low variance most likely due to the model making assumptions about the distribution of the data).
  * E.g., linear regression makes four assumptions about the data it is modeling.

> **NOTE:** Above generalization is of course not true for all problems (especially those with high imbalance datasets where we are concerned in learning the micro patterns that allows us to discriminate between certain less-populated values or with simple models that assume a high distribution of the data).

An alternative explanation of the above may be rephrased as following: when the complexity of the learning algorithm needs to increases to better fit our data (e.g., linear regression to random forests), there will be a trade of decreasing bias at the expense of increasing variance. Likewise, the simpler our learning algorithm needs to be, there wil be a trade of decreasing variance at the expense of bias. Both of these concepts can be seen as overfitting and undefitting the data:

* overfitting == (low bias, high variance)
  * $Trade(E[\hat{y}] \approx E[y]) = \text{High Variance}$
  * Model is able to approximate the expected value of the data very well but at the expense of increasing the spread (variance) of its predictions.
* underfitting == (low variance, high bias)
  * $Trade(min(var_{\hat{y}})) = \text{High Bias}$
  * Model is able to cluster its predictions tightly but at the expense of not approximating the true data as well.

Intuitively, this means that complex models in general are better able to approximate our data as it makes less assumptions (or restrictions) about our data. In the other hand, simpler models are able to reach more consensus (or precision) of its predictions due to their underlying assumptions.

> NOTE: Keep in mind that complex models are not necessarily inherently low bias and high variance. This depends on our data. If data has a high variance, then a complex model may be specially suited to model such distributions. However, this does not mean that a complex model is unable to fit low variance data (I'll leave this as an exercise to the reader by testing how well a random forest can fit on a linear dataset).

**Consequences of Bias-Variance Tradeoff? How can they be mitigated?**

As referenced above, the consequences of the bias-variance tradeoff is that no general model exists that will perform well on all types of datasets. Complex models such as random forests may perform very well in fitting data with high variance, but they also suffer from overfitting the data. On the other hand, simpler models like linear regression may fit approximately linear data well, but comes at the expense of possibly underfitting the data due to its many assumptions about the data its modeling. Resultingly, the only true method to find the best model that mitigates these consequences is through pure trial-and-error. We must experiment with a variety of models (from simple to complex) on the dataset and analyze where the bias-variance tradeoff is mitigated. This is a very expensive process in terms of computation as we can also tweak the parameters of models to find the best fit that minimizes error with a particular model. Due to such reasons and the added fact that a dataset may contain thousands of entries, it is imperative to find training schemes that minimize error while containing computation complexity from increasing exponentially. This is one of the reasons why training schemes like forward-stepwise selection is much more feasible than best-subset selection. 

