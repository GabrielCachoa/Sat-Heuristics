

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

def neighborhood(p,s):
  m=p[0]
  V=np.zeros((m+1,m))
  V[0,:]=s
  for i in range(1,m+1):
    V[i,:] = s
    V[i,i-1] = np.abs(1-V[i,i-1])
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


def rcl(P,L,S,alpha):

  S_copy = np.copy(S)
  GreedyValue = [greedy_f(P,S+[L[i]]) for i in range(len(L))]

  Max = np.max(GreedyValue)
  Min = np.min(GreedyValue)
  value = Max - alpha * (Max - Min)

  CandidatesRestrictedList = []
  for i in range(len(L)):
    if GreedyValue[i] > value:
      CandidatesRestrictedList.append(L[i])

  return CandidatesRestrictedList


def SolutionFormat(p,S):

  s = np.array([1 if i+1 in S else 0 for i in range(p[0])])
  
  return s

def GreedyRandomizedConstruction(P,p,alpha):
  S = []
  L = candidates(p,S)
  GreedyValue = [greedy_f(P,[L[i]]) for i in range(len(L))]

  while len(L) != 0:
    
    RCL_list = rcl(P,L,S,alpha)

    if len(RCL_list) != 0:
      index = random.randint(0,len(RCL_list))
      Choosen = RCL_list[index]
      S.append(Choosen)
      L = candidates(p,S)

    else:
      GreedyValue = [greedy_f(P,[L[i]]) for i in range(len(L))]
      Choosen = L[np.argmax(GreedyValue)]
      S.append(Choosen)
      L = candidates(p,S)

  s_star = SolutionFormat(p,S)

  return s_star

def grasp(P,p,alpha,TIME):
  start_time = time.time()
  now_time = time.time()

  f_star = 0

  while now_time - start_time < TIME:

    s0 = GreedyRandomizedConstruction(P,p,alpha)

    V = neighborhood(p,s0)
    (f_s1,s1) = local_search(P,V)

    if f_s1 > f_star:
      s_star = np.copy(s1)
      f_star = f_s1

    now_time = time.time()
  
  return (f_star,s_star)



if __name__ == "__main__":
    import sys
    (p,P) = load(sys.argv[1])
    s = random.randint(0,2,p[0])
    TIME = float(sys.argv[2])
    print(grasp(P,p,0.2,TIME))