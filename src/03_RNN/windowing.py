import torch

x = torch.arange(10, 100, 10)
print(x)  # tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
idx = torch.tensor([0, 1, 2])
print(idx)  # tensor([0, 1, 2])
print(x[idx])  # tensor([10, 20, 30])
idx = torch.tensor([
    [0, 1, 2],
    [1, 2, 3]
])
print(x[idx])  # tensor([[10, 20, 30],
               #         [20, 30, 40]])
n_time = 3
time = torch.arange(0, n_time).reshape(1, -1)
print(time)  # tensor([[0, 1, 2]])

# ---
# Explanation:
# - torch.arange(start, end) creates a 1D tensor with values from start up to (but not including) end.
# - Here: torch.arange(0, 3) → tensor([0, 1, 2]).
# 
# .reshape(1, -1)
# - .reshape(rows, cols) changes the shape of the tensor.
# - 1 means: make it 1 row.
# - -1 is a placeholder that tells PyTorch: "figure out the correct number of columns automatically."
# - Since the tensor has 3 elements, PyTorch reshapes it into shape (1, 3).
# ---


n_time = 3
n_window = len(x) - n_time + 1
print(n_window)  # 7

window = torch.arange(0, n_window).reshape(-1, 1)
print(window)  # tensor([[0],
        	   #	     [1],
        	   #	     [2],
        	   #		 [3],   
        	   #		 [4],
        	   #		 [5],
        	   #		 [6]])
print(time.shape, window.shape)  # (torch.Size([1, 3]), torch.Size([7, 1]))
idx = time + window
print(idx)  # tensor([[0, 1, 2],
        	#		  [1, 2, 3],
        	#		  [2, 3, 4],
        	#		  [3, 4, 5],
        	#		  [4, 5, 6],
        	#		  [5, 6, 7],
        	#		  [6, 7, 8]])
print(x[idx])  # tensor([[10, 20, 30],
        	   #	     [20, 30, 40],
        	   #	     [30, 40, 50],
        	   #	     [40, 50, 60],
        	   #	     [50, 60, 70],
        	   #	     [60, 70, 80],
        	   #	     [70, 80, 90]])


Tensor = torch.Tensor
def window(x: Tensor, n_time: int) -> Tensor:
    """
    Generates and index that can be used to window a timeseries.
    E.g. the single series [0, 1, 2, 3, 4, 5] can be windowed into 4 timeseries with
    length 3 like this:

    [0, 1, 2]
    [1, 2, 3]
    [2, 3, 4]
    [3, 4, 5]

    We now can feed 4 different timeseries into the model, instead of 1, all
    with the same length.
    """
    n_window = len(x) - n_time + 1
    time = torch.arange(0, n_time).reshape(1, -1)
    window = torch.arange(0, n_window).reshape(-1, 1)
    idx = time + window
    return idx

# Now we can easily change the window size to for example 6:

idx = window(x, 6)
print("idx")
print(idx)
print("x[idx]")
print(x[idx]) # tensor([[10, 20, 30, 40, 50, 60],
        	  #	        [20, 30, 40, 50, 60, 70],
        	  #	        [30, 40, 50, 60, 70, 80],
        	  #	        [40, 50, 60, 70, 80, 90]])

# Will this scale to more dimensions?
# Let's imagine we have 5 timesteps, every timestep 3 features being observed. 
# We can organize that into a `(5x3)` matrix instead of the one-dimensional raw data we started this notebook with.

x = torch.randint(0, 10, (5, 3))
print(x)  # tensor([[7, 8, 9],
          #	        [9, 8, 9],
          #	        [0, 7, 9],
          #	        [9, 2, 1],
          #	        [7, 7, 5]])

# ---
# Explanation:
# torch.randint(low, high, size)
# - This function generates a tensor filled with random integers.
# - Integers are drawn uniformly from the range [low, high), meaning:
# - low is inclusive.
# - high is exclusive.
# So here: values will be between 0 and 9.
# This produces a 2D tensor with shape (5, 3) filled with random integers from 0 to 9
# ---

# Now lets window it in chunks of 4 timesteps:
idx = window(x, 4)
print(x[idx])  # tensor([[[7, 8, 9],
         	   #	 	  [9, 8, 9],
         	   #		  [0, 7, 9],
         	   #		  [9, 2, 1]],
               #
        	   #		 [[9, 8, 9],
         	   #		  [0, 7, 9],
         	   #		  [9, 2, 1],
         	   #		  [7, 7, 5]]])

# We can use the windowed index to generate three training examples, every training examples covering four timesteps.
# We can also apply this to batched. Let us have a batch (B) of 32 examples, every example having 6 timesteps (T) and 2 features (F). This is organized in a `(B, T, F)` matrix.
# We can now apply the window on the 0th example, and squueze out an additional 3 training examples

batch = torch.randint(0, 10, (32, 6, 2))  # Create 32 times a 6x2 matrix
print("batch")
print(batch)
x = batch[0]
print("x")
print(x)
# tensor([[8, 4],
#         [9, 0],
#         [8, 4],
#         [7, 3],
#         [8, 1],
#         [9, 5]])
idx = window(x, 4)  # Note x=6 (6 rows of 2) and therefore n_window=4
print("idx")
print(idx)
# tensor([[0, 1, 2, 3],  # Representing rows 0,1,2,3 of x
#         [1, 2, 3, 4],  # Representing rows 1,2,3,4 of x
#         [2, 3, 4, 5]]) # Representing rows 2,3,4,5 of x	
x_windowed = x[idx]
print("x[idx]")
print(x[idx])
# tensor([[[8, 4],  # Row 0
#          [9, 0],  # Row 1
#          [8, 4],  # Row 2
#          [7, 3]], # Row 3

#         [[9, 0],  # Row 1
#          [8, 4],  # Row 2
#          [7, 3],  # Row 3
#          [8, 1]], # Row 4 

#         [[8, 4],  # Row 2
#          [7, 3],  # Row 3
#          [8, 1],  # Row 4
#          [9, 5]]])# Row 5
print(x.shape, x_windowed.shape)  # (torch.Size([6, 2]), torch.Size([3, 4, 2]))

