

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


def greedy_f(P,S):
  (n,m)=P.shape
  v=np.zeros(n)
  Q=np.copy(P)
  for i in range(n):
    for j in range(m):
      if Q[i,j] in S:
        v[i]=1
        break
  values = sum(v)
  return values

def candidates(p,S):
  id_plus = lambda x : x
  id_minus = lambda x : -x

  Range = list(range(1,p[0]+1))

  for i in range(len(S)):
    Range.remove(np.abs(S[i]))

  L = [i(x) for x in Range for i in (id_plus, id_minus)]
  
  return L

def solution_format(p,S):

  s = np.array([1 if i+1 in S else 0 for i in range(p[0])])
  
  return s


def pheromones_initialization(p):

  Pheromones = np.full((2*p[0]+1,2*p[0]+1),0.5)

  L = candidates(p,[])

  for i in range(2*p[0]+1):
    for j in range(2*p[0]+1):
      
      if j > 0:
        Pheromones[0,j] = L[j-1]

      if i == j:
        Pheromones[i,j] = 0
      elif (i%2) == 0 and j == i-1:
        Pheromones[i,j] = 0
      elif (i%2) != 0 and j == i+1:
        Pheromones[i,j] = 0
    
    if i > 0:
      Pheromones[i,0] = L[i-1]


  return Pheromones

def probabilities_initialization(p):
  Probabilities = np.full((2*p[0]+1,2*p[0]+1),1 / (p[0]-2))

  L = candidates(p,[])

  for i in range(2*p[0]+1):
    for j in range(2*p[0]+1):
      
      if j > 0:
        Probabilities[0,j] = L[j-1]

      if i == j:
        Probabilities[i,j] = 0
      elif (i%2) == 0 and j == i-1:
        Probabilities[i,j] = 0
      elif (i%2) != 0 and j == i+1:
        Probabilities[i,j] = 0
    
    if i > 0:
      Probabilities[i,0] = L[i-1]
  return Probabilities

def choose(p,S,L,Probabilities):
 
  ALLCandidates = Probabilities[:,0]
  ActualPosition = np.where(ALLCandidates == S[-1])[0]
  RowOfProbabilities = Probabilities[ActualPosition,:]
  
  ProbabilitiesL = []

  for i in range(len(L)):
    index = np.where(ALLCandidates == L[i])[0][0]
    ProbabilitiesL.append(RowOfProbabilities[0,index])
  
  Sum = np.sum(ProbabilitiesL)
  
  NormalizedProbabilities = [p/Sum for p in ProbabilitiesL]

  Choosen = random.choice(L,p = NormalizedProbabilities, size = 1)[0]

  return Choosen

def pheromones_intensification(Pheromones,BestWalk,delta):
  ALLCandidates = Pheromones[:,0]
  for i in range(len(BestWalk)-1):
    index0 = np.where(ALLCandidates == BestWalk[i])[0][0]
    index1 = np.where(ALLCandidates == BestWalk[i+1])[0][0]

    Pheromones[index0,index1] = Pheromones[index0,index1] + delta
    
  return Pheromones

def probabilities_matrix(Probabilities,Pheromones):
  for i in range(1,Probabilities.shape[0]):
    Sum = np.sum(Pheromones[i,1:])
    for j in range(1,Probabilities.shape[1]):
      Probabilities[i,j] = Pheromones[i,j] / Sum
  
  return Probabilities

def ant_colony_optimization(P,p,k,rho,delta,TIME):

  start_time = time.time()
  now_time = time.time()


  Pheromones = pheromones_initialization(p)

  Probabilities = probabilities_initialization(p)


  f_star = 0

  while now_time - start_time < TIME and f_star < p[1]:

    Solutions = []
    Walks = []
    for i in range(5):
      S = []
      L = candidates(p,S)
      
      S.append(L[random.randint(len(L))])

      while len(L) != 0:
        Choosen = choose(p,S,L,Probabilities)
        S.append(Choosen)
        L = candidates(p,S)

      s = solution_format(p,S)
      Solutions.append(s)
      Walks.append(S)

    Values = [f(P,s) for s in Solutions]

    BestWalk = Walks[Values.index(max(Values))]

    s_star = Solutions[Values.index(max(Values))]
    f_star = max(Values)

    for i in range(1,Pheromones.shape[0]):
      for j in range(1,Pheromones.shape[1]):
        Pheromones[i,j] = Pheromones[i,j] * (1-rho) 


    Pheromones = pheromones_intensification(Pheromones,BestWalk,delta)

    Probabilities = probabilities_matrix(Probabilities,Pheromones)

    now_time = time.time()


  return (f_star,s_star)



if __name__ == "__main__":
    import sys
    (p,P) = load(sys.argv[1])
    s = random.randint(0,2,p[0])
    TIME = float(sys.argv[2])
    print(ant_colony_optimization(P,p,100,0.15,0.2,TIME))