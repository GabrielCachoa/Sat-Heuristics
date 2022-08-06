

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import timeit
import pandas 
import csv
import time
import glob
from os.path import isfile



def load(fn):
  clauses = np.zeros(3)
  with open(fn, 'r') as data:
    lines = data.readlines()
    p=[]
    clauses = []
    for line in lines:
      if len(line) < 2 or line[0] in ['c', '%', '0']:
        continue

      if line[0]=='p':
        problemline=line.split()
        p.extend([int(problemline[2]),int(problemline[3])])
      else:
        clause = [int(x) for x in line.split()[0:-1]]
        clauses.append([x for x in clause])

    P=np.array(clauses)

  return (p,P)


def f(P,s):
  (n,m)=P.shape
  values=np.zeros(n)
  Q=np.copy(P)
  for i in range(n):
    for j in range(m):
      if s[np.abs(Q[i,j])-1]==0:
        Q[i,j]=Q[i,j]*(-1)
    values[i]=np.sign(np.max(Q[i,:]))
  values[values==-1]=0
  return sum(values)

def neighborhood(p,s,n,c):
  m=p[0]
  V=np.zeros((n+1,m))
  V[0,:]=s
  for i in range(1,n+1):
    V[i,:] = s
    for j in range(c):
      c1=random.randint(0,m)
      V[i,c1]=1-V[i,c1]
  return V


def local_search(P,V):
  v=[]
  for i in range(V.shape[0]):
    v.append(f(P,V[i,:]))
  f_star=max(v)
  s_star=V[v.index(max(v))]
  return (f_star,s_star)


def shaking(P,p,s_star,k,Sizes):

  V = neighborhood(p,s_star,Sizes[k][0],Sizes[k][1])
  i = random.randint(0,V.shape[0])

  s_0 = V[i,:]

  return s_0

def variable_neighborhood_search(P,p,s,Sizes,TIME):
  start_time = time.time()
  now_time = time.time()
  
  s_star = np.copy(s)
  f_star = f(P,s_star)

  k_max = len(Sizes)

  k_counter =np.zeros(k_max)

  T = [[s_star]]

  while (now_time - start_time < TIME) and f_star < p[1]:
    k=0
    while k < k_max:

      if np.array_equal(T[-1][0],s_star):
        T[-1].append(k)
      else:
        T.append([s_star,0])

      s_0 = shaking(P,p,s_star,k,Sizes)
      (f_s1,s1) = local_search(P,neighborhood(p,s_0,Sizes[k][0],Sizes[k][1]))

      k_counter[k] = k_counter[k]+1

      if f_s1 > f_star:
        s_star = np.copy(s1)
        f_star = f_s1
        k = 0
      else:
        k = k + 1
    now_time = time.time()

  return (f_star,s_star)

if __name__ == "__main__":
    import sys
    (p,P) = load(sys.argv[1])
    s = random.randint(0,2,p[0])
    TIME = float(sys.argv[2])
    sizes = [[300,4],[300,3],[300,2],[275,2],[250,2],[225,2],[200,2],[100,1]]
    print(variable_neighborhood_search(P,p,s,sizes,TIME))