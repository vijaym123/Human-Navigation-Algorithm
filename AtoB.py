import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import numpy
import pdb
import pickle
import time 
import bisect


#Actual Algorithm starts here. Previous Modules are used for analysis purposes.

reinforce_time = 0
Degree_Node = None
NodeList = None

def createLookup(hotSpots, G):
	'''
	hotSpots: List of hotSpots in descending order of importance
	G: considered Graph
	Given the Graph G and the list of HotSpots, this function returns the lookup table for the shortest path between each of the hotSpots. The return value is a 2 dimensional matrix.
	'''
	hotSpotLookup = []
	for i in range(len(hotSpots)):
		hotSpotLookup.append( [] ) #Create an empty 2-d Array List
	for i in range(len(hotSpots)):
		for j in range(len(hotSpots)):
			#print hotSpots[i],hotSpots[j]
			hotSpotLookup[i].append(nx.shortest_path(G,hotSpots[i],hotSpots[j],False) ) #Load the actual Shortest Path as an element of the 2-d array created
	return hotSpotLookup

def findHit(G, A, B, pathA, pathB):
	'''
	G: Graph which is undergoing machine learning
	A: Vertex #1
	B: Vertex #2
	pathA: contains the drunkard walk starting from A
	pathB: contains the drunkard walk starting from B
	Takes 2 vertices A and B from a graph G. Takes a random walk starting from A and takes another random walk starting from B and simultaneously builds the paths. If an intersection is found, the path is established and the corresponding intersection is returned. pathA and pathB are also dynamically updated.
	'''
	walkerA = A #walkers(robots) that take a random walk from given vertices
	walkerB = B

	pathA.append(A) #actual paths that the walkers take
	pathB.append(B)

	while( True ):
		walkerAAdj = G.neighbors(walkerA) #Adjacent vertices for current vertex
		walkerAAdj.remove('flags')

		walkerBAdj = G.neighbors(walkerB)
		walkerBAdj.remove('flags')

		randA = random.choice(walkerAAdj) #select one node from the set of neighbors of walkerA and walkerB and assign them to randA and randB respectively
		randB = random.choice(walkerBAdj)
		
		pathA.append(randA) #add the randomly selected edge to the set
		if randA not in pathB: #if randA is already in pathB, then the intersection has occured and there is no need to append randB to pathB. If we append randB to pathB, then there is a chance that we might get two intersection points if randB is also in pathA.
			pathB.append(randB)

		if (randA not in pathB and randB not in pathA): #If the sets are disjoint, then there is no common point that both the walkers now. So, proceed one step further for the next loop
			walkerA = randA
			walkerB = randB
		else:
			break #this implies that the intersection has occured and the infinite while loop should exit. Now, pathA and pathB are the components of the 2-Raw Random Walk

	if pathA[-1] in pathB: #if the last element of pathA is in pathB, then randA must have been the intersection point. Hence, set hit = randA = pathA[-1]
		hit = pathA[-1]
	else:
		hit = pathB[-1] #if the last element of pathB is in pathA, then randB must have been the intersection point. Hence, set hit = randB = pathB[-1]

	return hit

def createPath(pathA, pathB, hit):
	'''
	pathA: drunkard WALK starting from A
	pathB: drunkard WALK starting from B
	hit: the point at which hit has occured.
	Given two paths pathA and pathB and the intersection point hit, then this function integrates them into a path and returns the path. This path may contain cycles and must be removed.
	'''
	Path = []

	Path.extend(pathA[:pathA.index(hit)]) #calculate the index of the hit point and append the nodes in pathA to Path, excluding the hit point
	Path.extend(pathB[pathB.index(hit)::-1]) #calculate the index of the hit point and append the nodes in pathB to Path, including the hit point and IN REVERSE DIRECTION
	return Path

def findPath(G,A,B):
	'''
	A: Vertex #1
	B: Vertex #2
	This function takes in 2 vertices A and B in a graph G. It finds a path from A to B through the method of random walks. It returns the path and the intersection node of the random walk. 
	'''
	
	pathA = []
	pathB = []

	hit = findHit(G, A, B, pathA, pathB) #Take a random walk and stop when an intersection occurs. Return the intersection point.
	Path = createPath(pathA, pathB, hit) #Create a path from A to B. This path may contain cycles too.
	Path = removeCycles(Path) #Remove all the cycles from the current path.
	G[hit]['flags'] += 1 #Flag the hit point at every stage, rather than only for the minimum path case
	return Path, hit

def updateWeights(G, P, hit):
	'''
	P: path from A to B, with cycles removed
	hit: the actual hit point in the path
	'''
	Alength = P.index(hit) #path from A to hitMin must be rewarded with 1/(length). Hence, Alength will contain the length of this path.
	Blength = (len(P) - 1) - Alength #(len(Pmin) - 1) gives the total number of edges in the complete path. Subtracting Alength from it will give the path length from B to hitMin
	for i in range(Alength): #Assigning rewards to all the edges along the path from A to hitMin
		G[P[i]][P[i+1]]['weight'] += 1.0/Alength #PMin[i] is the starting vertex of the ith egde. The edge terminates at PMin[i+1]
			
	for i in range(Blength): #Assigning rewards to all the edges along the path from B to hitMin
		G[ P[Alength+i+1] ][ P[Alength+i] ]['weight'] += 1.0/Blength #we just just need to add Alength to the index in order to get the index for the path from B to hitMin.
	#IMPORTANT: The direction is reversed here. That is, we are incrementing (j,i) pair instead of the (i,j) pair.


def createRealWorld(name):
	'''
	name: The name of the real world graph
	This function creates a .graph file from a .gml file, and runs the machine learning alogorithm on it.
	'''	
	global reinforce_time
	G = nx.read_gml(name + ".gml")
	print name + ".gml" + " file read"

	StrMap = {}
	for node in G.nodes():
		StrMap[node] = str(node)
	G = nx.convert.relabel_nodes(G,StrMap)
	print "Undergoing Machine Learning..."
	start = time.time()
	H = reinforce(G) #Enforce Machine Learning to generate a gml file of the learnt graph.
	finish = time.time()
	reinforce_time = finish - start       
	nx.write_gpickle(H,str(name) + '.gpickle')

def createScaleFreeNetwork(numOfNodes, degree):
	'''
	numOfNodes: The number of nodes that the scale free network should have
	degree: The degree of the Scale Free Network
	This function creates a Scale Free Network containing 'numOfNodes' nodes, each of degree 'degree'
	It generates the required graph and saves it in a file. It runs the Reinforcement Algorithm to create a weightMatrix and an ordering of the vertices based on their importance by Flagging.
	'''
	global reinforce_time
	G = nx.barabasi_albert_graph(numOfNodes, degree) #Create a Scale Free Network of the given number of nodes and degree
	StrMap = {}
	for node in G.nodes():
		StrMap[node] = str(node)
	G = nx.convert.relabel_nodes(G,StrMap)

	print "Undergoing Machine Learning..."

	start = time.time()
	H = reinforce(G) #Enforce Machine Learning to generate a gml file of the learnt graph.
	finish = time.time()
	reinforce_time = finish - start

	print "Machine Learning Completed..."
	filename = "SFN_" + str(numOfNodes) + "_" + str(degree) + '.gpickle' 
	nx.write_gpickle(H,filename)#generate a gpickle file of the learnt graph.
	print "Learnt graph Successfully written into " + filename

def createErdos(numOfNodes, edgeProb):
	'''
	numOfNodes: The number of nodes that the Eldish graph will have
	edgeProb: The probability of existance of an edge between any two vertices
	This function creates an Erdos Graph containing 'numOfNodes' nodes, with the probability of an edge existing between any two vertices being 'edgeProb'
	It generates the required graph and saves it in a file. It runs the Reinforcement Algorithm to create a weightMatrix and an ordering of the vertices based on their importance.
	'''
	global reinforce_time
	G = nx.erdos_renyi_graph(numOfNodes, edgeProb) #Creates an Eldish Graph of the given number of nodes and edge Probability
	if nx.is_connected(G):
		StrMap = {}
		for node in G.nodes():
			StrMap[node] = str(node)
		G = nx.convert.relabel_nodes(G,StrMap)

		print "Undergoing Machine Learning..."
		start = time.time()
		H = reinforce(G) #Enforce Machine Learning to generate a gml file of the learnt graph.
		finish = time.time()
		reinforce_time = finish - start
		print "Machine Learning Completed..."
		filename = "EG_" + str(numOfNodes) + "_" + str(edgeProb) + ".gpickle"
		nx.write_gpickle(H,filename)#generate a gpickle file of the learnt graph.
		print "Learnt graph Successfully written into " + filename
		

def createBridge(numOfNodes, edgeProb, bridgeNodes):
	'''
	numOfNodes: Number of nodes in the clustered part of the Bridge Graph
	edgeProb: Probability of existance of an edge between any two vertices.
	bridgeNodes: Number of nodes in the bridge
	This function creates a Bridge Graph with 2 main clusters connected by a bridge.
	'''
	global reinforce_time
	G1 = nx.erdos_renyi_graph(2*numOfNodes + bridgeNodes, edgeProb) #Create an ER graph with number of vertices equal to twice the number of vertices in the clusters plus the number of bridge nodes.
	G = nx.Graph() #Create an empty graph so that it can be filled with the required components from G1
	G.add_edges_from(G1.subgraph(range(numOfNodes)).edges()) #Generate an induced subgraph of the nodes, ranging from 0 to numOfNodes, from G1 and add it to G
	G.add_edges_from(G1.subgraph(range(numOfNodes + bridgeNodes,2*numOfNodes + bridgeNodes)).edges()) #Generate an induced subgraph of the nodes, ranging from (numOfNodes + bridgeNodes) to (2*numOfNodes + bridgeNodes)

	A = random.randrange(numOfNodes) #Choose a random vertex from the first component
	B = random.randrange(numOfNodes + bridgeNodes,2*numOfNodes + bridgeNodes) #Choose a random vertex from the second component

	prev = A #creating a connection from A to B via the bridge nodes
	for i in range(numOfNodes, numOfNodes + bridgeNodes):
		G.add_edge(prev, i)
		prev = i
	G.add_edge(i, B)
	
	StrMap = {}
	for node in G.nodes():
		StrMap[node] = str(node)
	G = nx.convert.relabel_nodes(G,StrMap)
	print "Undergoing Machine Learning..."
	start = time.time()
	H = reinforce(G) #Enforce Machine Learning to generate a gml file of the learnt graph.
	finish = time.time()
	reinforce_time = finish - start
	print "Machine Learning Completed..."
	filename = "BG_" + str(numOfNodes) + "_" + str(edgeProb) + "_" + str(bridgeNodes) + ".gpickle"
	nx.write_gpickle(H,filename)#generate a gpickle file of the learnt graph.
	print "Learnt graph Successfully written into " + filename

def reinforce(G):
	'''
	G: Graph G which has to undergo machine learning
	'''
	global Degree_Node

	Degree_Node = G.degree()
	H = nx.DiGraph(G)
	NodeList = H.nodes()
	EdgeList = H.edges()

	for A,B in EdgeList:
		H[A][B]['weight'] = 0

	for A in NodeList:
		H[A]['flags'] = 0
		#H[A]['learningFrac'] = 0

	learningFrac = {}
	for A in NodeList:
		learningFrac[A] = 0

	count = 0
	numOfTrials = 0.9 * len(NodeList) * ( len(NodeList) + 1) / 2

	for i in range(int(numOfTrials)):
		A = random.choice(NodeList)
		B = random.choice(NodeList)

		#print "progress:", float(count) / numOfTrials * 100, "%"
		if A != B:
			count += 1
			P, hit = findPath(H,A,B) #P contains an acyclic path from A to B. hit contains the corresponding hit point. While computing the path, the Flagger is also updated.
			updateWeights(H, P, hit) #updates the weight matrix by adding 1/(length) to the edge-weights of the corresponding paths
				
			for node in P:
				if node!=hit:					
					neighbors = H.neighbors(node)
					neighbors.remove('flags')
					maxWeight = 0
					totalWeight = 0
					for neighbor in neighbors:
						if H[node][neighbor]['weight'] > maxWeight:
							maxWeight = H[node][neighbor]['weight']
						totalWeight += H[node][neighbor]['weight']
					learningFrac[node] = maxWeight / totalWeight #* Degree_Node[node]
					#print node, learningFrac[node]


	ratios = learningFrac.values()
	print "Average: ", numpy.average(ratios)
	print "Standard Deviation: ", numpy.std(ratios)

	return H

def removeCycles(Path):
	'''
	Given a path, this function removes all the cycles and returns the acyclic path
	'''
	i = 0 #i is the walker
	while i < len(Path): #the length of the path keeps on decreasing as the control flow of the program progresses
		for j in range(len(Path) - 1, i,-1): #move j from the last position to the (i+1)th position, when the path[i] and path[j] are the same, this indicates a cycle. Hence, remove the nodes in between them (inclusive of either end).
			if Path[j] == Path[i]:
				del(Path[i:j])
				break
		i += 1
	return Path


def test(GLearnt,A,B,hotSpots,hotSpotLookup):
	'''
	This is the actual navigation phase of the algorithm. This algorithm has been modified a lot, with respect to the algorithm in the paper. Backtracking on encountering a deadend node (node from which any hop will lead to a cycle) is also implemented.
	'''
	walkerA = A #walkerA and walkerB are the robots that navigate through the network
	walkerB = B

	pathA = [A] #Locus of walkerA and walkerB initialized with A and B respectively. 
	pathB = [B]

   	backTrackA = []
	nodeWeights = []
	for node, attrib in GLearnt[A].items():
		try:
			nodeWeights.append((attrib['weight'],node))
		except TypeError:
			pass
	
	nodeWeights.sort(reverse=True)
	
	nodeWeights = [node for weight,node in nodeWeights]
	backTrackA.append(nodeWeights)
	
	backTrackB = []
	nodeWeights = []
	for node, attrib in GLearnt[B].items():
		try:
			nodeWeights.append((attrib['weight'],node))
		except TypeError:
			dummy = 'Do Nothing'
	nodeWeights.sort(reverse=True)
 	
	nodeWeights = [node for weight,node in nodeWeights]
	backTrackB.append(nodeWeights)

	while True: #Loop until a PATH (previous integration algo produced cycles as well) is found.
	#Take simultaneous greedy hops for walkerA and walkerB. After each hop, check whether the paths have intersected. If so, integrate the paths there itself. The hops for walkerA and walkerB are taken only till they hit the hotspot. This loop is exited when both walkerA and walkerB are in hotspots and pathA and pathB do not intersect.

		if walkerA not in hotSpots: #if walkerA is a hotspot itself, there is no need for continuing.
			#select that neighbor of walkerA with the maximum edge weight.

			maxNode = backTrackA[-1][0]

			while maxNode in pathA:
					#for neigh in backTrackA:
						#print neigh
					#raw_input("Press return to continue...")
					if backTrackA[-1] != []:
						backTrackA[-1].pop(0)
						maxNode = backTrackA[-1][0]
					else:
						backTrackA.pop()
						pathA.pop()
						walkerA = pathA[-1]
						backTrackA[-1].pop(0)
						maxNode = backTrackA[-1][0]
	
			walkerA = maxNode #change the state of walkerA
			pathA.append(maxNode) #append the next vertex to pathA

			nodeWeights = []
			for node, attrib in GLearnt[maxNode].items():
				try:
					nodeWeights.append((attrib['weight'],node))
				except TypeError:
					dummy = 'Do Nothing'

			nodeWeights.sort(reverse=True)
			nodeWeights = [node for weight,node in nodeWeights]
			backTrackA.append(nodeWeights)
			
			if walkerA in pathB: #check whether an intersection has occured. If so, integrate pathA and pathB(reversed). Note that hotspots do not come into the picture here.
				fullpath = pathA[:]
				pathBrev = pathB[:pathB.index(walkerA)]
				pathBrev.reverse()
				fullpath.extend(pathBrev)
				return fullpath


		if walkerB not in hotSpots: #if walkerB is a hotspot itself, there is no need for continuing.
			#select that neighbor of walkerB with the maximum edge weight.

			maxNode = backTrackB[-1][0]

			while maxNode in pathB:
					#for neigh in backTrackB:
						#print neigh
					#raw_input("Press return to continue...")
					if backTrackB[-1] != []:
						backTrackB[-1].pop(0)
						maxNode = backTrackB[-1][0]
					else:
						backTrackB.pop()
						pathB.pop()
						walkerA = pathB[-1]
						backTrackB[-1].pop(0)
						maxNode = backTrackB[-1][0]
	
			walkerB = maxNode #change the state of walkerA
			pathB.append(maxNode) #append the next vertex to pathA

			nodeWeights = []
			for node, attrib in GLearnt[maxNode].items():
				try:
					nodeWeights.append((attrib['weight'],node))
				except TypeError:
					dummy = 'Do Nothing'

			nodeWeights.sort(reverse=True)
			nodeWeights = [node for weight,node in nodeWeights]
			backTrackB.append(nodeWeights)
			
			if walkerB in pathA: #check whether an intersection has occured. If so, integrate pathA and pathB(reversed). Note that hotspots do not come into the picture here.
				fullpath = pathB[:]
				pathArev = pathA[:pathA.index(walkerA)]
				pathArev.reverse()
				fullpath.extend(pathArev)
				return fullpath

		if walkerA in hotSpots and walkerB in hotSpots: #while loop exits when both walkerA and walkerB are hotspots.
			break

	#This block is entered only when there is no intersection between pathA and pathB until both reach the hotspots. If there was an intersection, then this function would have returned the integrated path in the previous block itself.

	fullPath = pathA[:] #copy the complete path from A to hotSpot1 (including hotSpot1) to the fullPath
	fullPath.extend(hotSpotLookup[ hotSpots.index(pathA[-1]) ][ hotSpots.index(pathB[-1]) ][1:] ) #Lookup for the shortest Path between hotSpot1 and hotSpot2 and extend it to the list fullPath excluding the first element
	pathB.reverse() #pathB needs to be reversed before appending
	fullPath.extend(pathB[1:]) #append the path from hotSpot2 to b to the full path
	fullPath = removeCycles(fullPath) #A-hotSpot1, hotSpot1-hotSpot2, hotSpot2-B. Here, hotSpot1-hotSpot2 might contain the nodes in A-hotSpot1 and B-hotSpot2. Hence, they cause cycles. Therefore, they must be removed.
	return fullPath

def query(GLearnt, hotSpots, hotSpotLookup):
	i = 1
	global Degree_Node
	
	
	G = nx.Graph(GLearnt)
	for i in NodeList:
		Degree_Node[i] = [Degree_Node[i],GLearnt.neighbors(i)]	
	PlainAdamicFullPaths = []
	PCAFullPaths = []

	djk_time = 0
	PlainAdamic_time = 0
	PCA_time = 0
	
	total = len(NodeList) * len(NodeList)
	count = 0
	for A in NodeList:
		for B in NodeList:
			src = A#raw_input("Enter source name:")
			dstn = B#raw_input("Enter destination name:")

			start = time.time()
			PlainAdamicFullPath = testadamic(G,src,dstn)
			finish = time.time()
			PlainAdamic_time+=(finish-start)

			start = time.time()			
			PCAFullPath = test(G, src, dstn, hotSpots, hotSpotLookup)
			finish = time.time()
			PCA_time+=(finish-start)

			start = time.time()
			ShortestPath = nx.shortest_path(GLearnt, src, dstn)
			finish = time.time()
			djk_time+=finish-start

			PlainAdamicFullPaths.append(float(len(PlainAdamicFullPath))/len(ShortestPath))
			PCAFullPaths.append(float(len(PCAFullPath))/len(ShortestPath))

			count += 1
			print "Progress: ", float(count) / total

	print "Avg of PlainAdamicFullPaths :" , numpy.average(PlainAdamicFullPaths)
	print "Avg of PCAFullPaths :" , numpy.average(PCAFullPaths)
	print "PCA / PlainAdamic", numpy.average(PCAFullPaths) / numpy.average(PlainAdamicFullPaths)

	PCA_time += reinforce_time
	print "reinforce_time : ",reinforce_time
	print "PCA time : ", PCA_time
	print "djk time : ",djk_time
	print "adamic time : " ,PlainAdamic_time

	print "PlainAdamic_time/PCA_time:", PlainAdamic_time / PCA_time
	print "PlainAdamic_time/djk_time:", PlainAdamic_time / djk_time

def FindAlpha(Flags, Threshold):
	'''Given a list called Flags, this method returns the optimal alpha, i.e the number of hotspots. We use the maximum sum method here.'''
	
	maxLen = 0
	maxPoint = -1
	last = int((len(Flags) - 1)*Threshold)
	for i in range(1,last):
		Len = math.sqrt( math.pow(i-0,2) + math.pow(Flags[i] - Flags[0],2) ) + math.sqrt( math.pow(last - i,2) + math.pow(Flags[i] - Flags[last],2) )
		if Len > maxLen:
			maxLen = Len
			maxPoint = i
	#print "Cut Point based on Sum:", maxPoint
	return maxPoint

def Navigate(GLearnt):
	'''
	GLearnt: The Machine Learnt Graph - networkx object
	'''
	global NodeList

	Flagger = [] #Flagger is a list of tuples, containing the flag value for the corresponding nodes.
	NodeList = GLearnt.nodes()
	for node in NodeList:
		Flagger.append((GLearnt[node]['flags'],node))
	Flagger.sort(reverse=True) 	#Reverse the Flagger List. Hence, its now in descending order.
	Flags = [f for f,v in Flagger]
	Alpha = FindAlpha(Flags, 0.2)
	hotSpots = [v for f,v in Flagger[:int(Alpha)]] #Choose the top alpha nodes, as hotspots.
	for i in GLearnt.nodes():
		del(GLearnt[i]['flags'])
	hotSpotLookup = createLookup(hotSpots,GLearnt) #Create a Lookup Table for the hotSpot List. This table will also work for directed graphs. Hence, a 2-d matrix is the best Data Structure.
	
	#Navigation Phase starts here.
	query(GLearnt, hotSpots, hotSpotLookup)

def adamicwalk(G,a,b,path_a,path_b):
	global Degree_Node	

	path_a.append(a)
	path_b.append(b)		

	Flagger_a = {}
	Flagger_b = {}

	while True :
		maxdegree = -1
		maxnode = None
		neighbors_a = Degree_Node[a][1]
		for i in neighbors_a:
			if(maxdegree<Degree_Node[i][0]):
				maxdegree=Degree_Node[i][0]
				maxnode=i

		while Flagger_a.has_key(maxnode):
			try:
				maxnode = neighbors_a.pop() #the neighbor with the next highest degree is NOT being chosen here.
			except IndexError:
				print "Empty =>> Exiting abruptly ...don't know how to handle this case....yet"
				exit()
		path_a.append(maxnode)
		Flagger_a[maxnode] = 1 

		if Flagger_b.has_key(maxnode):
			return maxnode

		a = maxnode

		neighbors_b = Degree_Node[b][1]
		maxdegree = -1
		maxnode = None	
		for i in neighbors_b:
			if(maxdegree<Degree_Node[i][0]):
				maxdegree=Degree_Node[i][0]
				maxnode=i
	
		while Flagger_b.has_key(maxnode):
			try:
				maxnode = neighbors_b.pop() #the neighbor with the next highest degree is NOT being chosen here.
			except IndexError:
				print "Empty =>> Exiting abruptly ...don't know how to handle this case....yet"
				exit()
		path_b.append(maxnode)
		Flagger_b[maxnode] = 1
		
		if Flagger_a.has_key(maxnode):
			return maxnode
		b = maxnode

def testadamic(G,a,b):
	path_a = []
	path_b = []
	hit = adamicwalk(G,a,b,path_a,path_b)
	return createPath(path_a, path_b, hit)				

