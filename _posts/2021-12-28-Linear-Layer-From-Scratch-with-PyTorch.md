---

layer: splash
title: "Linear Layer From Scratch with PyTorch"
author: Erick Platero
category: DL
tags: [dl] 

---

# Linear Layer From Scratch with PyTorch



**Forward/Backward pass implementation?**

```python
# NOTE: `@staticmethod` simply let's us initiate a class without instantiating it
import torch
from torch import nn
from typing import Optional, Tuple

class Linear_Layer(torch.autograd.Function):
    """
    obj: implement forward/backward pass of linear layer 
    """
    @staticmethod
    def forward(ctx, input: torch.FloatTensor,weights: torch.FloatTensor, 
                bias: Optional[bool] = None) -> torch.FloatTensor:
        """
        :obj: implement forward pass of linear layer
        :param input: input matrix; shape: (B, in_dim)
        :param weights: weight matrix; shape: (in_dim, out_dim)
        :param bias: whether to include bias or not; shape: (out_dim) if not None
        """
        
        # save input, weight, and bias for backward pass
        ctx.save_for_backward(input, weights, bias)
        
        # linear transformation
        output = torch.mm(input, weights) # shape: (B, out_dim) = (B, in_dim) * (in_dim, out_dim)
        
        # if bias is applied
        if bias is not None:
            # add bias to each element of `output`
            expanded_bias = bias.unsqueeze(0).expand_as(output) # shape: (B, out_dim), repeats bias B times
            output += expanded_bias
        
        return output

    
    @staticmethod
    def backward(ctx, incoming_grad: torch.FloatTensor) \
                 -> Tuple[Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        :obj: implement backward pass of linear layer
        :param incoming_grad: gradient of loss w.r.t. `output` from forward pass; shape: (B, out_dim)
        """
        
        # extract inputs from forward pass
        input, weights, bias = ctx.saved_tensors 
        
        # assume none of the extracted inputs need gradients
        grad_input = grad_weight = grad_bias = None
        
        # if input requires grad
        if ctx.needs_input_grad[0]:
            grad_input = incoming_grad.mm(weights.t()) # shape: (B, in_dim) = (B, out_dim) * (out_dim, in_dim)
            
        # if weights require grad
        if ctx.needs_input_grad[1]:
            grad_weight = incoming_grad.t().mm(input) # shape: (out_dim, in_dim) = (out_dim, B) * (B, in_dim) 
            
        # if bias requires grad
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = incoming_grad.sum(0) # shape: (out_dim,)
        
        
        # add grad_output.t() to match original layout of weight parameter
        return grad_input, grad_weight.t(), grad_bias
```



**Implement linear layer API:**

```python
class Linear(nn.Module):
    """
    :obj: implement linear layer API 
    """
    def __init__(ctx, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        
        # init weight parameter
        ctx.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        
        # if `bias` is applied
        if bias:
            # init bias parameter
            ctx.bias = nn.Parameter(torch.randn((out_dim)))
        else:
            # register parameter as None 
            ctx.register_parameter("bias",None)
        
    def forward(ctx, input: torch.FloatTensor) -> torch.FloatTensor:
        output = Linear_Layer.apply(input, ctx.weight, ctx.bias)
        return output
```



**Assert results match PyTorch's official implementation:**

```python
x = torch.randn((20, 3)) # init pseudo-input
pt_linear = nn.Linear(3, 5, bias = False) # init pytorch linear layer
man_linear = Linear(3,5, bias = False) # init manual linear layer

# make weights of `pt_linear` equal weights of `man_linear`
pt_weight = pt_linear.weight.t() # pytorch weight orientation equals manual orientation once transposed
man_linear.weight = nn.Parameter(pt_weight)

# test forward pass
pt_out = pt_linear(x)
man_out = man_linear(x)
assert torch.equal(pt_out, man_out)

# execute pytorch linear backward pass
pt_pred = pt_out.sum()
pt_pred.backward()
pt_weight_grad = pt_linear.weight.grad.t()

# execute manual linear backward pass
man_pred = man_out.sum()
man_pred.backward()
man_weight_grad = man_linear.weight.grad

# test backward pass
assert torch.equal(pt_weight_grad, man_weight_grad)
```