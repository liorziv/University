import networkx as nx
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

class Graph(object):

	def __init__(self, edges_file = None):

		self.__G = nx.DiGraph()
		self.__predecessors = {}

		if edges_file:
			self.__load_from_file(edges_file)


	def __load_from_file(self, edges_file):
		edgesSet = self.__read_file(edges_file)

		for edge in edgesSet:
			if (edge[0] != edge[1]):
				self.__G.add_edge(int(edge[0]), int(edge[1]), weight=int(edge[2]))

				if int(edge[1]) not in self.__predecessors:
					self.__predecessors[int(edge[1])] = []

				if int(edge[0]) not in self.__predecessors:
					self.__predecessors[int(edge[0])] = []

				self.__predecessors[int(edge[1])].append((int(edge[0]), int(edge[2])))


	def __read_file(self, file_path):
		edges = []

		with open(file_path, "rb") as fl:
			reader = csv.reader(fl, delimiter='\t')
			for row in reader:
				edges.append(row)
		return  edges


	def get_nodes(self):
		return self.__G.nodes()


	def get_nodes_to_neighbours(self):
		
		nodes_to_neighbours_dct = {}

		for src, dest in self.__G.edges():

			if src not in nodes_to_neighbours_dct:
				nodes_to_neighbours_dct[src] = []

			nodes_to_neighbours_dct[src].append((dest, self.__G.edge[src][dest]["weight"]))

		return nodes_to_neighbours_dct


	def get_communities(self, k=4):
		
		community_to_researcher_list_dct = {}

		undirected_graph = self.__G.to_undirected()

		for community_id, researcher_list in enumerate(nx.k_clique_communities(undirected_graph, k)):
			community_to_researcher_list_dct[community_id] = [int(val) for val in researcher_list]

		return community_to_researcher_list_dct


	def get_in_edges(self, node_id):
		return self.__predecessors[node_id]


	def compute_local_clustering_coefficient(self):
		coef_list = {}

		g_nodes = self.__G.nodes()

		# holds the number of incoming edges to each node
		inDict = self.__G.in_degree(g_nodes)

		for node in g_nodes:
			tmp = 0
			# saves the number of neighboors - the out edges
			ki = len(self.__G.neighbors(node))

			# avoid zero partition
			if(ki > 1):
				tmp = float(inDict[node] + ki) / (ki * (ki - 1))

			# insert the coffiecent to a dict
			coef_list[node] = tmp

		return coef_list


if __name__ == "__main__":

	G = Graph("tests/numbered.edges")

	community_count_list = []

	for i in range(2, 10):
		communities = G.get_communities(k=i)
		
		try:
			community_count_list.append(np.log10(len(communities)))

		except BaseException:
			community_count_list.append(0)


	plt.plot([i for i in range(2, 10)], community_count_list)
	plt.title("Amount of communities per k-value")
	plt.xlabel("k value used for clique perculation")
	plt.ylabel("log amount of communities found")
	plt.show()

	for community_id, researcher_list in G.get_communities(k=4).items():
		print community_id, researcher_list
