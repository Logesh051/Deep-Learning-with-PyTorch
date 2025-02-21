#!/usr/bin/env python
# coding: utf-8

# # PyTorch Basics Exercises
# For these exercises we'll create a tensor and perform several operations on it.
# 
# <div class="alert alert-danger" style="margin: 10px"><strong>IMPORTANT NOTE!</strong> Make sure you don't run the cells directly above the example output shown, <br>otherwise you will end up writing over the example output!</div>

# ### 1. Perform standard imports
# Import torch and NumPy

# In[1]:


import torch
import numpy as np



# ### 2. Set the random seed for NumPy and PyTorch both to "42"
# This allows us to share the same "random" results.

# In[2]:


np.random.seed(42)
torch.manual_seed(42)

print("Random seed set to 42 for both NumPy and PyTorch!")


# ### 3. Create a NumPy array called "arr" that contains 6 random integers between 0 (inclusive) and 5 (exclusive)

# In[3]:


import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a NumPy array with 6 random integers between 0 and 5 (5 is exclusive)
arr = np.random.randint(0, 5, 6)

print("NumPy Array:", arr)


# In[3]:


# DON'T WRITE HERE


# ### 4. Create a tensor "x" from the array above

# In[6]:


import torch
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create NumPy array
arr = np.random.randint(0, 5, 6)

# Convert NumPy array to PyTorch tensor
x = torch.tensor(arr)  # or use torch.from_numpy(arr)
print(x)


# In[4]:


# DON'T WRITE HERE


# ### 5. Change the dtype of x from 'int32' to 'int64'
# Note: 'int64' is also called 'LongTensor'

# In[7]:


# Convert x to int64 (LongTensor)
x = x.to(torch.int64)  # or x = x.long()

print("Updated Tensor:", x)
print("Tensor dtype:", x.dtype)


# In[5]:


# DON'T WRITE HERE


# ### 6. Reshape x into a 3x2 tensor
# There are several ways to do this.

# In[9]:


# Reshape x into a 3x2 tensor
x_reshaped = x.reshape(3, 2)  # or x.view(3, 2)

print(x_reshaped)


# In[6]:


# DON'T WRITE HERE


# ### 7. Return the right-hand column of tensor x

# In[12]:


# Get the right-hand column as a column vector
right_column = x_reshaped[:, 1].unsqueeze(1)  # Adds a second dimension

print(right_column)


# In[7]:


# DON'T WRITE HERE


# ### 8. Without changing x, return a tensor of square values of x
# There are several ways to do this.

# In[15]:


# Compute the square of x without modifying it, keeping the 3x2 shape
x_squared = x_reshaped ** 2  # or use torch.pow(x_reshaped, 2)

print(x_squared)


# In[8]:


# DON'T WRITE HERE


# ### 9. Create a tensor "y" with the same number of elements as x, that can be matrix-multiplied with x
# Use PyTorch directly (not NumPy) to create a tensor of random integers between 0 (inclusive) and 5 (exclusive).<br>
# Think about what shape it should have to permit matrix multiplication.

# In[17]:


import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a 2x3 tensor of random integers between 0 (inclusive) and 5 (exclusive)
y = torch.randint(0, 5, (2, 3))

print(y)


# In[9]:


# DON'T WRITE HERE


# ### 10. Find the matrix product of x and y

# In[19]:


# Compute matrix multiplication
result = torch.matmul(x_reshaped, y)  # or use x_reshaped @ y

print(result)


# In[10]:


# DON'T WRITE HERE


# ## Great job!
