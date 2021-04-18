from AntColonyOptimizer import AntColonyOptimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import time
import warnings
import sympy
np.random.seed(0)

dist = sympy.randMatrix(r= 100 ,c = 100, min = 1, max =99, symmetric= True, seed = 0)
# sympy 의 인덱스 접근은 [r][c] 형식이 아닌 [r,c]형식 

for i in range(100):
    dist[i,i] = 0

dist = np.array(dist,dtype='float64')
print(dist)
optimizer = AntColonyOptimizer(ants=100, evaporation_rate=.1, intensification=2, alpha=1, beta=1,beta_evaporation_rate=0, choose_best=.1)

best = optimizer.fit(dist,100)
optimizer.plot()
