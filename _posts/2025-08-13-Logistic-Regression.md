---
layout: single
classes: wide
title: "Logistic-Regression"
author: Erick Platero
category: DL
tags: [dl]
author_profile: true

---
# Logistic Regression

Logistic Regression is a simple yet powerful classification technique used to analyze the relationship between a quantitative variable $x$ and a dichotomous categorical variable $y$. Similar to linear regression, this relationship is established by applying a linear transformation to the input data, which can be represented as:

$$
Y = Xw
$$

This equation forms the foundation of Logistic Regression, where $Y$ is the output, $X$ is the input feature matrix, and $w$ represents the weight parameters that need to be learned during training.

# Binary-Cross-Entropy

However, Logistic Regression is unique in the way that it **learns relationships.** Given that we have to classify y given x, a natural choice of performance/criterion is the Binary Cross Entropy Loss.

In short, Cross Entropy is a popular choice for evaluating classification models whose output is a probability distribution. It is computed by applying the following function to each prediction:

$$
L(\hat{y},y)=y\cdot -log(\hat{y})^T
$$

For binary classification tasks, Cross Entropy is reformulated as:

$$
L(\hat{y},y)=y\cdot-log(\hat{y})+(1-y)\cdot(-log(1-\hat{y}))
$$

In general, Cross-Entropy loss increases exponentially as the predictions diverge from the actual labels. Conversely, as the predictions become closer to the true labels, the loss approaches zero. The graph below illustrates the Binary Cross-Entropy loss for targets of 1 and 0 at various prediction levels:

<img src="{{ site.baseurl }}/assets/logistic/BCE.png" alt="Binary Cross-Entropy Loss Graph" style="zoom:75%;" />

To compute the gradient of the Loss function with respect to the predictions during the backward pass, we apply basic calculus:

$$
\begin{split}\frac{\partial L(\hat{y},y)}{\partial \hat{y}} & =\frac{\partial}{\hat{y}}(y\cdot-log(\hat{y})) + \frac{\partial}{\hat{y}}(1-y)\cdot(-log(1-\hat{y})) \\& =\frac{-y}{\hat{y}} - (\frac{1-y}{1-\hat{y}}* -1) \\& = \frac{-y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\end{split}
$$

Given that the Loss function is usually the last forward operation, it also becomes the first gradient we need to compute for the backward pass. For this reason, there is no incoming gradient that we need to worry about integrating.

However, instead of calculating a backward pass of *each prediction* w.r.t. our weight parameters, we usually take the mean prediction confidence as a way to better gauge our model's performance.  

In this case, we will be calculating below gradients

$$
\frac{\partial }{\partial \hat{w}}avg(L(\hat{y},y))
$$


Once our loss has been  computed, we follow the general DL procedure of taking a "step" towards steepest descent by computing the gradient of our Loss/criterion function w.r.t. weight parameters

$$
w_j=w_j-\alpha\frac{\partial }{\partial w_j}L(w_j)
$$

## Sigmoid Layer

To utilize Binary Cross-Entropy as our loss criterion, we need to ensure that our model's output is within the range ```[0,1]```. This is achieved by applying a ```Sigmoid``` layer before feeding the outputs to the Loss function. The Sigmoid layer acts as an activation function that maps any real-valued number to a value between 0 and 1, effectively "squeezing" the input into the desired range.

The Sigmoid function is defined as:

$$
\sigma(y)=\frac{1}{1+e^{-y}}
$$

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png" alt="Sigmoid function - Wikipedia" style="zoom:40%;" />

A key property of the Sigmoid function is that its derivative can be expressed in terms of its output, simplifying the computation during backpropagation:

$$
\frac{\partial \sigma}{\partial y} = \sigma(y)(1-\sigma(y))
$$

Since the Sigmoid function is applied element-wise, its derivative is computed independently for each input element:

$$
\sigma(y) = \sigma\begin{pmatrix}y_1 & y_2 &y_3\end{pmatrix} = \begin{pmatrix}\sigma(y_1) & \sigma(y_2) & \sigma(y_3)\end{pmatrix}
$$
$$
\frac{\partial \sigma}{\partial y} = \begin{pmatrix}\sigma(y_1)(1-\sigma(y_1) & \sigma(y_2)(1-\sigma(y_2) &\sigma(y_3)(1-\sigma(y_3)\end{pmatrix}
$$

Given that the Sigmoid layer does not introduce any new parameters, its backward pass is considered an intermediate operation. The gradient of the loss with respect to the input of the Sigmoid layer can be obtained by integrating the incoming gradient from the loss ($\frac{\partial L}{\partial \sigma}$) with the derivative of the Sigmoid function ($\frac{\partial \sigma}{\partial y}$) through a Hadamard product:

$$
\frac{\partial L}{\partial y}=\frac{\partial L}{\partial \sigma}\odot \frac{\partial \sigma}{\partial y}
$$

For a deeper understanding of these concepts, refer to the [Linear Layer]() and [ReLU]() tutorials.

## Implementing Logistic Regression

With the necessary components defined, we will now manually implement the forward and backward passes for each operation using PyTorch.

**Note:** The implementation details of the Linear Layer are not covered here as they were discussed in the [Linear Layer tutorial](Linear Layer.ipynb).



```python
import torch
import torch.nn as nn
####################### Linear Layer ###################


class Linear_Layer(torch.autograd.Function):
    """
    Define a Linear Layer operation
    """
    @staticmethod
    def forward(ctx, input,weights, bias = None):
        """
        In the forward pass, we feed this class all necessary objects to 
        compute a  linear layer (input, weights, and bias)
        """
        # input.dim = (B, in_dim)
        # weights.dim = (in_dim, out_dim)
        
        # given that the grad(output) wrt weight parameters equals the input,
        # we will save it to use for backpropagation
        ctx.save_for_backward(input, weights, bias)
        
        
        # linear transformation
        # (B, out_dim) = (B, in_dim) * (in_dim, out_dim)
        output = torch.mm(input, weights)
        
        if bias is not None:
            # bias.shape = (out_dim)
            
            # expanded_bias.shape = (B, out_dim), repeats bias B times
            expanded_bias = bias.unsqueeze(0).expand_as(output)
            
            # element-wise addition
            output += expanded_bias
        
        return output

    
    @staticmethod
    def backward(ctx, incoming_grad):
        """
        In the backward pass we receive a Tensor (output_grad) containing the 
        gradient of the loss with respect to our f(x) output, 
        and we now need to compute the gradient of the loss
        with respect to our defined function.
        """
        # incoming_grad.shape = (B, out_dim)
        
        # extract inputs from forward pass
        input, weights, bias = ctx.saved_tensors 
        
        # assume none of the inputs need gradients
        grad_input = grad_weight = grad_bias = None
        
        
        # if input requires grad
        if ctx.needs_input_grad[0]:
            # (B, in_dim) = (B, out_dim) * (out_dim, in_dim)
            grad_input = incoming_grad.mm(weights.t())
            
        # if weights require grad
        if ctx.needs_input_grad[1]:
            # (out_dim, in_dim) = (out_dim, B) * (B, in_dim) 
            grad_weight = incoming_grad.t().mm(input)
            
        # if bias requires grad
        if bias is not None and ctx.needs_input_grad[2]:
            # torch.ones((1,B)).mm(incoming_grad)  
            # (out) = (1,B)*(B,out_dim)
            grad_bias = incoming_grad.sum(0)
        
        
        # below, if any of the grads = None, they will simply be ignored
        
        # add grad_output.t() to match original layout of weight parameter
        return grad_input, grad_weight.t(), grad_bias
        
        
```


```python
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # define parameters
        
        # weight parameter
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        
        # bias parameter
        if bias:
            self.bias = nn.Parameter(torch.randn((out_dim)))
        else:
            # register parameter as None if not initialized
            self.register_parameter('bias',None)
        
    def forward(self, input):
        output = Linear_Layer.apply(input, self.weight, self.bias)
        return output
```


```python
################## Sigmoid Layer #######################

# Remember that our incoming gradient will be of equal dims as our output
# b/c of this, output now becomes an intermediate variable
# input.shape == out.shape == incoming_gradient.shape

import torch.nn as nn
import torch

class sigmoid_layer(torch.autograd.Function):
    
    def __init__(self):
        ''
    
    def sigmoid(self,x):
        sig = 1 / (1 + (-1*x).exp())
        return sig
    
    # forward pass
    def forward(self, input):
        # save input for backward() pass 
        self.save_for_backward(input) 
        activated_input = self.sigmoid(input)
        return activated_input

    # integrate backward pass with incoming_grad
    def backward(self, incoming_grad):
        """
        In the backward pass we receive a Tensor containing the 
        gradient of the loss with respect to our f(x) output, 
        and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        chained_grad = (self.sigmoid(input) * (1- self.sigmoid(input))) * incoming_grad
        return chained_grad
```


```python
# test forward pass

weight = torch.tensor([1.], requires_grad = True)
input = torch.tensor([1.])
x = input * weight
sig = sigmoid_layer()(x)
sig # tensor([0.7311], grad_fn=<sigmoid_layer>)
```




```python
# test backward pass

sig.backward(torch.tensor([1.]))
weight.grad # tensor([0.1966])
```




```python
# compare output with PyTorch's inherent Method

weight = torch.tensor([1.], requires_grad = True)
input = torch.tensor([1.])
x = input * weight

sig = nn.Sigmoid()(x)
sig # tensor([0.7311], grad_fn=<SigmoidBackward>)
```





```python
sig.backward()
weight.grad # tensor([0.1966])
```



```python
# Wrap ReLU_layer function in nn.module

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

        
    def forward(self, input):
        output = sigmoid_layer()(input)
        return output
    
```


```python
####################  Binary Cross Entropy ################

# inputs must all be of type .float()
class BCE_loss(torch.autograd.Function):
    

    @staticmethod
    def forward(self, yhat, y):
        # save input for backward() pass 
        self.save_for_backward(y,yhat) 
        loss = - (y * yhat.log() + (1-y)* (1-yhat).log())
        return loss

    @staticmethod
    def backward(self, output_grad):
        y,yhat = self.saved_tensors
        chained_grad = ((yhat-y) / (yhat * (1- yhat)))
        
        # y does not need gradient and thus we pass None to signify this
        return chained_grad, None
```


```python
# test above method
output = torch.tensor([.50], requires_grad = True)
y = torch.tensor([1.])
loss = BCE_loss.apply(output,y)
loss # tensor([0.6931], grad_fn=<BCE_lossBackward>)
```




```python
# test backward() method
loss.backward()
output.grad # tensor([-2.])
```




```python
# test with PyTorch
output = torch.tensor([.50], requires_grad = True)
y = torch.tensor([1.])
bce = nn.BCELoss()
loss = bce(output,y)
loss # tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
```



```python
# test backward() method
loss.backward()
output.grad # tensor([-2.])
```



```python
# Wrap BCELoss function in nn.module

class BCELoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super().__init__()

        
    def forward(self, pred, target):
        output = BCE_loss.apply(pred,target)
        # reduce output by average
        output = output.mean()
        return output
    
```

Now that we have all of our "ingredients", we can now create our Logistic model


```python
# Create Logistic Regression function
class LogisticRegression(nn.Module):
    def __init__(self, input_dim = 30):
        super().__init__()
        self.linear = Linear(input_dim, 1) 
        self.sigmoid = Sigmoid()
        
    def forward(self,x):
        # output.shape = (B, 1)
        output = self.sigmoid(self.linear(x))
        return output.view(-1)
```

## Wisconsin Breast Cancer Dataset

To demonstrate the effectiveness of Logistic Regression, we will train our model on the Wisconsin Breast Cancer Dataset to classify cancer cells as either malignant or benign based on the characteristics of their nuclei. For more information about the dataset, visit the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).


```python
# import data
import pandas as pd
url = 'https://raw.githubusercontent.com/PacktPublishing/Machine-Learning-with-R-Second-Edition/master/Chapter%2003/wisc_bc_data.csv'
df = pd.read_csv(url)
df.index = df.diagnosis
df.drop(columns = ['diagnosis','id'],inplace = True)
df.head()
```



```python
df.info(verbose = True)
```

```
<class 'pandas.core.frame.DataFrame'>
Index: 569 entries, B to M
Data columns (total 30 columns):
radius_mean          569 non-null float64
texture_mean         569 non-null float64
perimeter_mean       569 non-null float64
area_mean            569 non-null float64
smoothness_mean      569 non-null float64
compactness_mean     569 non-null float64
concavity_mean       569 non-null float64
points_mean          569 non-null float64
symmetry_mean        569 non-null float64
dimension_mean       569 non-null float64
radius_se            569 non-null float64
texture_se           569 non-null float64
perimeter_se         569 non-null float64
area_se              569 non-null float64
smoothness_se        569 non-null float64
compactness_se       569 non-null float64
concavity_se         569 non-null float64
points_se            569 non-null float64
symmetry_se          569 non-null float64
dimension_se         569 non-null float64
radius_worst         569 non-null float64
texture_worst        569 non-null float64
perimeter_worst      569 non-null float64
area_worst           569 non-null float64
smoothness_worst     569 non-null float64
compactness_worst    569 non-null float64
concavity_worst      569 non-null float64
points_worst         569 non-null float64
symmetry_worst       569 non-null float64
dimension_worst      569 non-null float64
dtypes: float64(30)
memory usage: 137.8+ KB
```



```python
# visualize the distribution of our binary classes

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sns.countplot(df.index);plt.show()
```


<img src="{{ site.baseurl }}/assets/logistic/class_distr.png" alt="class distribution" style="zoom:75%;" />


Given that there is about twice more data on Benign cells than Malignant, a model will become bias towards classifying Benign cells if not properly addressed.

## Data Preprocessing




```python
# Separate features (X) from target (y)
import numpy as np

X = df.values
y = (df.index == 'M')
y = y.astype(np.double)
```


```python
# normalize features
from sklearn.preprocessing import normalize
X = normalize(X, axis = 0)
```


```python
# parse data to training and testing set for evaluation

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state = 42, shuffle = True)
```


```python
# Transform data to PyTorch tensors and separate data into batches
from skorch.dataset import Dataset
from torch.utils.data import DataLoader

# Wrap each observation with its corresponding target
train = Dataset(X_train,y_train) 
test = Dataset(X_test,y_test) 

# separate data into batches of 16
train_dl = DataLoader(train, batch_size = 16, pin_memory = True)
test_dl = DataLoader(test, batch_size = 16, pin_memory = True)
```

Now that we have all the data formatted, let's instatiate our model, criterion, and optimizer

# Instantiate Logistic Regression


```python
# instantiate model and place it on GPU

device = torch.device('cuda')
model = LogisticRegression(30).to(device)
model
```

```
LogisticRegression(
  (linear): Linear()
  (sigmoid): Sigmoid()
)
```




```python
# initiate loss function
criterion = BCELoss()
```


```python
# initiate optimizer
from torch import optim
optimizer = optim.SGD(model.parameters(), lr = .01)
```

Make one forward pass to make sure everything works as it should


```python
# test train_dl
batch_X,batch_y = next(iter(train_dl))
print(f"batch_X.shape: {batch_X.shape}")
print('-'*35)
print(f"batch_X.shape: {batch_y.shape}")
```

```
batch_X.shape: torch.Size([16, 30])
-----------------------------------
batch_X.shape: torch.Size([16])
```



```python
# Assert our model makes as many predictions according to the batch
# all inputs must be of type .float()

output = model(batch_X.cuda().float())
output.shape
```




    torch.Size([16])




```python
# average loss
loss = criterion(output,batch_y.cuda().float())
loss # tensor(0.6896, device='cuda:0', grad_fn=<MeanBackward0>)
```



```python
# compute gradients by calling .backward()
loss.backward()
```


```python
# take a step
optimizer.step()
```

# Train Logistic Regression

Now that we have asserted our model works as should, it's time to train it


```python

def train(model, iterator, optimizer, criterion):
    
    # hold avg loss and acc sum of all batches
    epoch_loss = 0
    epoch_acc = 0
    
    
    for batch in iterator:
        
        # zero-out all gradients (if any) from our model parameters
        model.zero_grad()
        
        
        
        # extract input and label
        
        # input.shape = (B, fetures)
        input = batch[0].cuda().float()
        # label.shape = (B)
        label = batch[1].cuda().float()
        
        
        # Start PyTorch's Dynamic Graph
        
        # predictions.shape = (B)
        predictions = model(input)
        
        # average batch loss 
        loss = criterion(predictions, label)
        
        # calculate grad(loss) / grad(parameters)
        # "clears" PyTorch's dynamic graph
        loss.backward()
        
        
        # perform SGD "step" operation
        optimizer.step()
        
        
        # Given that PyTorch variables are "contagious" (they record all operations)
        # we need to ".detach()" to stop them from recording any performance
        # statistics
        
        
        # average batch accuracy
        acc = binary_accuracy(predictions.detach(), label)
        

        
        # record our stats
        epoch_loss += loss.detach()
        epoch_acc += acc
        
    # NOTE: tense.item() unpacks Tensor item to a regular python object 
    # tense.tensor([1]).item() == 1
        
    # return average loss and acc of epoch
    return epoch_loss.item() / len(iterator), epoch_acc / len(iterator)

```


```python
# compute average accuracy per batch

def binary_accuracy(preds, y):
    # preds.shape = (B)
    # y.shape = (B)

    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).sum()
    acc = correct.item() / len(y)
    return acc
```


```python
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    
```


```python
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
        
    # turn off grad tracking as we are only evaluation performance
    with torch.no_grad():
    
        for batch in iterator:

            # extract input and label       
            input = batch[0].cuda().float()
            label = batch[1].cuda().float()


            # predictions.shape = (B,1)
            predictions = model(input)

            # average batch loss 
            loss = criterion(predictions, label)

            # average batch accuracy
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss
            epoch_acc += acc
        
    return epoch_loss.item() / len(iterator), epoch_acc / len(iterator)
```


```python
N_EPOCHS = 150

# track statistics
track_stats = {'epoch': [],
               'train_loss': [],
              'train_acc': [],
              'valid_loss':[],
              'valid_acc':[]}


best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_dl, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_dl, criterion)
    
    end_time = time.time()
    
    # record operations
    track_stats['epoch'].append(epoch + 1)
    track_stats['train_loss'].append(train_loss)
    track_stats['train_acc'].append(train_acc)
    track_stats['valid_loss'].append(valid_loss)
    track_stats['valid_acc'].append(valid_acc)
    
    

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # if this was our best performance, record model parameters
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_log_regression.pt')
    
    # print out stats
    print('-'*75)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

```

```
---------------------------------------------------------------------------
Epoch: 01 | Epoch Time: 0m 0s
  Train Loss: 0.662 | Train Acc: 62.65%
    Val. Loss: 0.653 |  Val. Acc: 63.28%
---------------------------------------------------------------------------
Epoch: 02 | Epoch Time: 0m 0s
  Train Loss: 0.658 | Train Acc: 62.65%
    Val. Loss: 0.649 |  Val. Acc: 63.28%
---------------------------------------------------------------------------
Epoch: 03 | Epoch Time: 0m 0s
  Train Loss: 0.655 | Train Acc: 62.65%
    Val. Loss: 0.646 |  Val. Acc: 63.28%
---------------------------------------------------------------------------
Epoch: 04 | Epoch Time: 0m 0s
  Train Loss: 0.652 | Train Acc: 62.65%
    Val. Loss: 0.643 |  Val. Acc: 63.28%
---------------------------------------------------------------------------
Epoch: 05 | Epoch Time: 0m 0s
  Train Loss: 0.649 | Train Acc: 62.65%
    Val. Loss: 0.639 |  Val. Acc: 63.28%
---------------------------------------------------------------------------
Epoch: 150 | Epoch Time: 0m 0s
  Train Loss: 0.409 | Train Acc: 87.50%
    Val. Loss: 0.393 |  Val. Acc: 90.62%
```


# Visualization

Our model performed very well! With a top validation accuracy of 90.62%

Now, let us graph our results


```python
# format data 
import pandas as pd

stats = pd.DataFrame(track_stats)
stats
```




```python
# format data
data = []
for row in stats.iterrows():
    data.append(row[1].to_dict())
# plot data
import hiplot as hip
hip.Experiment.from_iterable(data).display(force_full_width = True)
```

<iframe src="{{ site.baseurl }}/assets/logistic/logistic_training_hiplot.html" width="100%" height="500px"></iframe>



The above graph gives us a very nice way to visualize our expected general patterns: 
as the number of epoch increases, train and validation loss decreases while train and validation accuracy increase

Further, we can investigate the magnitude of each weight parameter to shed insight on the variables that had a higher level of influence on our prediction (assuming that higher magnitudes correlate with higher importance)

**NOTE**: this is only possible as our model is just a one layer linear operation. If this was a "deep" model, interpretation by weight magnitude would not be possible


```python
params = list(model.parameters())[0].detach().cpu().view(-1).numpy()
param_df = pd.DataFrame(params).T
param_df.columns = df.columns
param_df
```






```python
plt.figure(figsize = (4,8))
sns.heatmap(param_df.T.sort_values(0,ascending = False))
```






    
<img src="{{ site.baseurl }}/assets/logistic/heatmap.png" alt="logistic" style="zoom:75%;" />


Assuming that weights with higher magnitudes equate to a higher level of importance, we can see that 

1. area_worst
2. area_se and
3. points_worst

were the top 3 variables that helped us differentiate between Malign and Benign cells.

# Conclusion

Logistic Regression is a powerful method for analyzing the relationship between quantitative and binary qualitative variables, leveraging deep learning techniques to establish meaningful relationships.

Due to its "shallow" architecture compared to other deep learning models, Logistic Regression requires less data to learn effective representations.

Therefore, Logistic Regression is a valuable concept for anyone in the field of Data Science to understand and apply.
