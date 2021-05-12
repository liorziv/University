from operator import itemgetter
import sys
import numpy as np
import itertools

import itertools

K = 4



#generates a graph
class GraphGeneretor:

    def __init__(self,V,E):

        self.vertexesDicNames = {}
        self.vertexesDicIdx = {}
        self.V = V
        self.neighboorMat = np.zeros((len(V),len(V)))
        for index,v in enumerate(V):
            self.vertexesDicNames[v] = index
            self.vertexesDicIdx[index] = v

        self.E = E
        for node1,node2 in E :
            i = self.vertexesDicNames[node1]
            j = self.vertexesDicNames[node2]
            self.neighboorMat[i, j] = 1
            self.neighboorMat[j, i] = 1



    #creates a graph out of a text file
    def returnGraphSets(fileName):
        V = []
        E = []
        inputFile = open(fileName,'r')
        currLine = inputFile.readline()
        #reads the file until EOF
        while(currLine != ''):
            [node1,node2] = currLine.strip().split(' ')
            #adds a new vertex
            V.append(int(node1))
            V.append(int(node2))
            #check that it is not a self loop
            if (node1 != node2):
                E.append((int(node1),int(node2)))
            currLine = inputFile.readline()
        #truns the list to uniqe values
        V = set(V)

        #vertices = list(set(vertices))
        return V,E



    #removes all the parallel edges
    def removeParallelEdges(self):
        #runs on edges from E and sort them
        for edge1 in self.E:
            edge1S = sorted(edge1)
            for edge2 in self.E:
                edge2S = sorted(edge2)
                if(edge1S == edge2S and edge1 != edge2 ):
                    self.E.remove(edge2)
                    continue
        return len(self.E)



    #finds connected componentes
    def runCC(self):
        visited = []
        group = []
        CC = []
        for node in range (len(self.V)):
            if(node not in visited):
                group.append(node)
                newComp = set()
                while(len(group) > 0):
                    newNode = int(group.pop())
                    if(newNode not in visited):
                        visited.append(newNode)
                        newComp.add(newNode)
                        for node2Idx,isNeibhoor in enumerate(self.neighboorMat[newNode,:]):
                            if(isNeibhoor):
                                group.append(node2Idx)
                CC.append(newComp)
        return CC


    #finds if a given group is a clique
    def isclique(self,nodesList):
        for node1 in nodesList:
            for node2 in nodesList:

                if((node1 != node2) and (self.neighboorMat[self.vertexesDicNames[node1],self.vertexesDicNames[node2]] == 0)):
                    return False

        return True

    #find if two group have k-1 common vetexes(perloaction)
    def isPercolation(self,c1,c2):
        count = 0
        for node1 in c1:
            if(node1 in c2):
                count = count + 1
        return count == K-1

    #percolation algorithem
    def percolation(self):
        combList = itertools.combinations(self.V,K)
        cliqueList = []
        perlocationList = []
        for group in combList:
            if(self.isclique(group)):
                cliqueList.append(group)

        for c1 in cliqueList:
            for c2 in cliqueList:
                if(c1 != c2 and self.isPercolation(c1,c2)):
                    if(sorted((c1,c2)) not in perlocationList):
                        perlocationList.append(sorted((c1,c2)))




        return perlocationList,cliqueList;

#translates into the original group format
def tranlation(cc,perGraph):
    translationList = []
    totList = []

    for group in cc:
        tmplist = set()
        for elem in group:
            for var in perGraph.vertexesDicIdx[elem]:
                tmplist.add(var)
       totList.append(sorted(tmplist))

    return totList

#finds the max coneccted component
def maxCC(CC):
    max = 0
    for comp in CC:
        if(len(comp) > max):
            max = len(comp)
    return max



def main():

    fileName = sys.argv[1]
    [V, E] = GraphGeneretor.returnGraphSets(fileName)
    graph = GraphGeneretor(V,E)
    print(r'The number of (vertexs,edges) in the graph: %d, %d' % (len(graph.vertexesDicNames), graph.removeParallelEdges()))
    CC1 = graph.runCC()
    print(r'The number of connected components in the graph: %d' % len(CC1))
    print(r'The max connected component: %d' % maxCC(CC1))
    perlocationList, cliqueList = graph.percolation()
    perlocationGraph = GraphGeneretor(cliqueList,perlocationList)
    CC2 = perlocationGraph.runCC()
    for lst in tranlation(CC2,perlocationGraph):
        print(lst)
    print(r'The number of  communities in graph: %d'% len(CC2))

if __name__ == "__main__":
    main()
