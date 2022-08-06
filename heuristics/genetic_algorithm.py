

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


def initialization(P,p,size):

  Population = []
  for i in range(size):
    Population.append(random.randint(0,2,p[0]))
  Population=np.array(Population)

  return Population

def selection(P,p,Population,size):

  a=np.arange(Population.shape[0])
  MatingPool=[]
    
  Values = [f(P,Population[i,:]) for i in range(Population.shape[0])]
  sum = np.sum(Values)
  P = [value/sum for value in Values]

  for i in range(int(size/5)):
    MatingPool.append(random.choice(a,2,p))

  return MatingPool

def recombination(P,p,MatingPool,Population):
  Offspring=[]

  for pair in MatingPool:
    n=random.randint(p[0])
    son=np.append(Population[pair[0],0:n],Population[pair[1],n:p[0]])
    Offspring.append(son)
  return Offspring

def mutation(P,p,Offspring):

  M=random.uniform(0,1,len(Offspring))
  M[M > 0.2]=0
  M[M != 0]=1

  for i in range(len(Offspring)):
    if M[i]==1:
      l=Offspring[i].shape[0]
      n=random.randint(l)
      Offspring[i][n]=1-Offspring[i][n]

  return Offspring

def replacement(P,p,Population,Offspring,size):
  new_population=[]
  for i in range(size):
    new_population.append(np.append(Population[i,:],f(P,Population[i,:])))

  new_population=np.array(new_population)
  new_population = new_population[np.argsort(new_population[:, -1])]
  new_population=np.delete(new_population,-1,1)
  new_population=np.delete(new_population,range(int(size/5)),0)

  for son in Offspring:
    new_population=tuple(new_population)
    new_population=np.vstack((new_population,son))
    
  new_population=np.array(new_population)
  return Population

def genetic_algorithm(P,p,size,TIME):
  
  start_time = time.time()
  now_time = time.time()

  Population = initialization(P,p,size)

  f_star = 0
  while now_time - start_time < TIME and f_star < p[1]:


    MatingPool = selection(P,p,Population,size)

    Offspring = recombination(P,p,MatingPool,Population)

    Offspring = mutation(P,p,Offspring)

    Population = replacement(P,p,Population,Offspring,size)

    now_time = time.time()
  
    (f_star,s_star) = local_search(P,Population)

  return (f_star,s_star) 


if __name__ == "__main__":
    import sys
    (p,P) = load(sys.argv[1])
    s = random.randint(0,2,p[0])
    TIME = float(sys.argv[2])
    print(genetic_algorithm(P,p,100,TIME))