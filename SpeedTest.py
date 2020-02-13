import time
import numpy as np
a=[[i for i in range(1000)]for j in range(1000)]
t0=time.time()
for i in range(1000):
    for j in range(1000):
        a[i][j]=2
t1=time.time()
print(str(t1-t0))