import numpy
import math
import AtoB
import matplotlib.pyplot as plt
import random
import networkx as nx
import pdb
import pickle



numOfNodes = 100
degree = 3
edgeProb = 0.3

AtoB.createErdos(numOfNodes,edgeProb)
G = nx.read_gpickle("EG_" + str(numOfNodes) + "_" + str(edgeProb) + ".gpickle")

#AtoB.createScaleFreeNetwork(numOfNodes, degree)
#G = nx.read_gpickle("SFN_" + str(numOfNodes) + "_" + str(degree) + ".gpickle")
AtoB.Navigate(G)
