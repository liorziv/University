from Graph import Graph
import pickle
import sys
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import csv

class ProjectRunner(object):

	def __init__(self, edges_file, researcher_name_file):

		self.__G = Graph(edges_file)

		with open(researcher_name_file, "rb") as fl:
			self.__researcher_name_to_id_dct = {}

			reader = csv.reader(fl, delimiter="\t")

			for row in reader:
				self.__researcher_name_to_id_dct[row[0]] = int(row[1])


	def run(self):
		
		# Get common data structures
		communities_dct = self.__G.get_communities()
		print "Community count:", len(communities_dct)

		# Analyze the graph
		print "Finding Enriching communities"
		enriching_communnities_list = \
			self.extract_enriching_communities(communities_dct)

		# enriching_communnities_list = [i for i in range(960)]

		print "Run statistical tests"
		self.analyze_effect_over_researcher(enriching_communnities_list,
								   			communities_dct)

		return 0


	def extract_enriching_communities(self, community_to_researcher_dct):

		node_to_neighbours_dct = self.__G.get_nodes_to_neighbours()
		node_to_coef_dct = self.__G.compute_local_clustering_coefficient()
		dest_to_source_dct = self.__G.get_nodes_to_neighbours()

		community_to_citation_prob_dct = self.__calculate_citation_prob(dest_to_source_dct,
															  community_to_researcher_dct)

		self.__plot_histogram_from_dct_citation_probability(community_to_citation_prob_dct, "intra-citation probabiliy")

		researcher_to_cc_dct =  self.__G.compute_local_clustering_coefficient()

		community_to_cc_dct = self.__calculate_avg_local_cluster_coefficient(community_to_researcher_dct, researcher_to_cc_dct)
		self.__plot_histogram_from_dct_cc(community_to_cc_dct, param_str = "clustering coefficient")

		enriched_communities = self.__find_enriched_communities(1, 0.67, community_to_cc_dct, community_to_citation_prob_dct)

		print "Enriching communities:"

		for en in enriched_communities:
			print en, community_to_researcher_dct[en]

		return enriched_communities
		

	def analyze_effect_over_researcher(self, enriching_communnities_list, communities_dct):
		
		self.__plot_researchers_rank(communities_dct)

		threshold = 4 

		# Go over the communities and extract the amount
		researchers_to_ranks_dct = self.__calculate_researchers_ranks()

		rich_in_enriching = 0
		rich_in_not_enriching = 0
		not_rich_in_enriching = 0
		not_rich_in_not_enriching = 0
		enriched_rank_list = []
		not_enriched_rank_list = []


		# Go over the researchers
		for researcher, researcher_id in self.__researcher_name_to_id_dct.items():
			researcher_community = -1

			# if researchers_to_ranks_dct[researcher_id] >= threshold:
				# print researcher_id, "is rich", researchers_to_ranks_dct[researcher_id], type(researchers_to_ranks_dct[researcher_id])

			# Find the researcher community
			for community_id, researcher_list in communities_dct.items():
				if researcher_id in researcher_list:
					researcher_community = community_id
					break

			# if its enriching
			if researcher_community in enriching_communnities_list:
				enriched_rank_list.append(researchers_to_ranks_dct[researcher_id])

				if researchers_to_ranks_dct[researcher_id] >= threshold:
					rich_in_enriching += 1

				else:
					not_rich_in_enriching += 1

			else:
				not_enriched_rank_list.append(researchers_to_ranks_dct[researcher_id])

				if researchers_to_ranks_dct[researcher_id] >= threshold:
					rich_in_not_enriching += 1

				else:
					not_rich_in_not_enriching += 1

		print "rich_in_enriching", rich_in_enriching
		print "rich_in_not_enriching", rich_in_not_enriching
		print "not_rich_in_enriching", not_rich_in_enriching
		print "not_rich_in_not_enriching", not_rich_in_not_enriching

		print stats.fisher_exact([[rich_in_enriching, rich_in_not_enriching], 
								  [not_rich_in_enriching, not_rich_in_not_enriching]])


		print "median", np.median(enriched_rank_list), np.median(not_enriched_rank_list)
		plt.boxplot([enriched_rank_list, not_enriched_rank_list])
		plt.ylim([-20, 20])
		plt.show()


	def __plot_researchers_rank(self, communities_dct):
		# First analyze citation distribution
		citation_count_list = []

		for node_id in self.__G.get_nodes():

			incoming_edges = self.__G.get_in_edges(node_id)
			
			if len(incoming_edges) > 0:
				_, weight_list = zip(*incoming_edges)
				citation_count_list.append(sum(weight_list))

				# if (citation_count_list[-1] >= 40):
				# 	print "researcher", node_id

				# 	for community_id, researcher_list in communities_dct.items():
				# 		if int(node_id) in researcher_list:
				# 			print "community_id", community_id
				# 			print researcher_list
				# 			break

				# 	print "*" * 20


		# print citation_count_list

		plt.hist(citation_count_list, bins=[i for i in range(0, 200)], log=True)
		plt.xlabel("Amount of citations")
		plt.ylabel("Log amount of researchers")
		plt.show()

		plt.plot([i for i in range(50, 100)], [np.percentile(citation_count_list, i) for i in range(50, 100)])
		plt.xlabel("Percentile")
		plt.ylabel("percentile citation value")
		plt.show()

	def __calculate_researchers_ranks(self):
		researcher_rank_dct = {}

		for researcher_id in self.__researcher_name_to_id_dct.values():
			incoming_edges = self.__G.get_in_edges(researcher_id)

			if len(incoming_edges) > 0:
				_, weight_list = zip(*incoming_edges)
				researcher_rank_dct[researcher_id] = sum(weight_list)

			else:
				researcher_rank_dct[researcher_id] = 0

		return researcher_rank_dct

	def __calculate_citation_prob(self, dest_to_source_dct, community_to_researcher_dct):
		"""
		:param dest_to_source_dct: dict of cited researcher to a list of its citers,
			   each citer is a list: researcher id and number of citations
		:param community_to_researcher_dct: dict of community id to list of researcher ids
		:return: dict of community ids to probabilities of a citation in it to be an intra-citation
		"""
		community_to_citation_prob_dct = {}

		for community in community_to_researcher_dct:
			inter_counter = 0.0
			intra_counter = 0.0

			# go over each researcher in the community
			for researcher in community_to_researcher_dct[community]:

				# go over the cited researchers and check wether "researcher" is a citer of dest
				# if so, see if dest is in the community and increment the appropriate counter
				for dest in dest_to_source_dct:
					source_lst = dest_to_source_dct[dest]

					for source in source_lst:
						citer = source[0]
						weight = source[1]

						# if the researcher in the community has cited dest
						if researcher == citer:
							# dest is the community- source cited a researcher inside the community
							if dest in community_to_researcher_dct[community]:
								intra_counter += weight

							else:  # dest is outside the community- source cited a researcher outside the community
								inter_counter += weight

			if inter_counter + intra_counter != 0.0:
				community_to_citation_prob_dct[community] = float(intra_counter)/ (inter_counter + intra_counter)
			else:
				community_to_citation_prob_dct[community] = 0.0

		return community_to_citation_prob_dct


	def __calculate_avg_local_cluster_coefficient(self, community_to_researcher_dct, researcher_to_cc_dct):
		"""
		:param community_to_researcher_dct: dict of community id to list of resercher ids
		:param researcher_to_cc_dct: dict of researcher id to its clustering coefficient
		:return: dict of community id to the avg cc of its researchers
		"""
		community_to_cc_dct = {}

		for community_id, researcher_list in community_to_researcher_dct.items():
			community_to_cc_dct[community_id] = \
				np.mean([researcher_to_cc_dct[researcher_id]
						 for researcher_id in researcher_list])

		return community_to_cc_dct


	def __plot_histogram_from_dct_citation_probability(self, dct, param_str):
		"""
		plots the "citetion level" and local avrage cluster coefficient in order to help us find a linear seperator
		:param dct: dict of community id to the avg cc of its researchers
		:param param_str: the subject of the histogram
		"""

		values = dct.values()
		# print values
		# fold = int(10 ** np.floor(np.log10(len(values)/100)))
		fold = 10
		bins = [float(i)/fold for i in range(fold + 1)]
		# print bins
		plt.figure()
		plt.hist(values, bins = bins)
		plt.xlim([0,1])
		plt.xlabel("ranges of " + param_str)
		plt.ylabel("number of cummunities in each " + param_str + " range")
		plt.savefig("number of cummunities in each " + param_str + " range")
		plt.show()


	def __plot_histogram_from_dct_cc(self, dct, param_str):
		"""
		plots the "citetion level" and local avrage cluster coefficient in order to help us find a linear seperator
		:param dct: dict of community id to the avg cc of its researchers
		:param param_str: the subject of the histogram
		"""

		values = dct.values()
		# print values
		# fold = int(10 ** np.floor(np.log10(len(values)/100)))
		fold = 10
		bins = [(5*float(i))/fold for i in range(fold + 1)]
		# print bins
		plt.figure()
		plt.hist(values, bins = bins)
		plt.xlabel("ranges of " + param_str)
		plt.ylabel("number of cummunities in each " + param_str + " range")
		plt.savefig("number of cummunities in each " + param_str + " range")
		plt.show()


	def __find_enriched_communities(self, coef_thershold, innerCitation_thershold, community_to_cc_dct, community_to_citation_prob_dct):
		"""
		finds only the enriched communities
		:param thershold: number between 0 to 1
		:param communitiesDec: community to cc
		:return:
		"""
		only_enriched = []
		for (key,value) in community_to_cc_dct.items():
			if(value >= coef_thershold and community_to_citation_prob_dct[key] >= innerCitation_thershold):
				only_enriched.append(key)

		return only_enriched
		# return community_to_cc_dct.keys()


if __name__ == "__main__":

	runner = ProjectRunner(edges_file=sys.argv[1], researcher_name_file=sys.argv[2])

	exit(runner.run())
