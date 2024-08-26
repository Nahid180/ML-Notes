import numpy as np
import pandas as pd

w_true = 2.5
b_true = 1.3
x = np.random.randint(0,10,50)
y = (w_true*x) + b_true

data_frame = pd.DataFrame({"X":x,"Y":y})
#lr = 0.001

#df= data_frame

def Gradient(df,lr,step):
    init_w = 0
    init_b = 0

    for i in range(step):
      temp_w = init_w - lr*((1/df.shape[0])*np.sum((init_w*df["X"] + init_b) - df["Y"]*df["X"]))
      temp_b = init_b - lr*((1/df.shape[0])*np.sum((init_w*df["X"] + init_b) - df["Y"]))
      init_w = temp_w
      init_b = temp_b

    return init_b, init_w

w,b=Gradient(data_frame,0.001,50)
print(f"Weight: {w} Bias: {b}")


