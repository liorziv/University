import requests
import xml.etree.ElementTree as ET
import cPickle as pickle
import sys
import glob
import os

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def generate_article_dict(article_list):
	return {int(article.get_pubmed_id()): article for article in article_list}


def generate_article_to_qouting_articles(article_dct):

	article_to_quouting_dct = {}

	j = 1

	for article_id_list in chunks([str(val) for val in article_dct.keys()], 30000):

		print "processing iteration"
		print j
		j = j + 1

		if os.path.exists("citation_%d.xml" % j):
			with open("citation_%d.xml" % j, "rb") as fl:
				html = fl.read()

		else:

			url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
			data = [("dbfrom", "pubmed"), ("linkname", "pubmed_pubmed_citedin")] + [("id", i) for i in article_id_list]
			response = requests.post(url=url, data=data)
			html = response.text

			with open("citation_%d.xml" % j, "wb") as fl:
				fl.write(html)

		for article_id in article_id_list:
				article_to_quouting_dct[int(article_id)] = []

		root = ET.fromstring(html)

		for link_set in root.findall("LinkSet"):
			id_list = link_set.find("IdList")
			article_id = int(id_list.find("Id").text)

			for link_set_db in link_set.findall("LinkSetDb"):
				for link in link_set_db.findall("Link"):
					for quoting_article_id in link.findall("Id"):
						article_to_quouting_dct[article_id].append(int(quoting_article_id.text)) 

	return article_to_quouting_dct


def generate_researchers_edges(article_dct, article_to_quouting_dct):

	edge_dct = {}

	# Go over each pair of articles (article and quoting article)
	for article_id, article in article_dct.items():
		for quoting_article_id in article_to_quouting_dct[article_id]:

			if quoting_article_id not in article_dct:
				continue

			# Go over all the authors and quoting authors
			for author in article.get_researchers():
				for quoting_authors in article_dct[quoting_article_id].get_researchers():

					if quoting_authors.get_full_name() == author.get_full_name():
						continue

					edge = (quoting_authors.get_full_name(), author.get_full_name())

					if edge not in edge_dct:
						edge_dct[edge] = 0
							
					edge_dct[edge] += 1

	return edge_dct


if __name__ == "__main__":

	workdir = sys.argv[1]

	article_list = []

	for i, article_file_path_list in enumerate(chunks(glob.glob("%s/*.bin" % workdir), 80)):
		print article_file_path_list

		article_list = []

		for article_file_path in article_file_path_list:

			with open(article_file_path, "rb") as fl:
				article_list.extend(pickle.load(fl))

		with open("all_articles_%d.article" % i, "wb") as fl:
			pickle.dump(article_list, fl)

		break

	article_dct = generate_article_dict(article_list)

	article_to_quouting_dct = generate_article_to_qouting_articles(article_dct)

	edge_dct = generate_researchers_edges(article_dct, article_to_quouting_dct)

	with open("all_edges.edges", "wb") as fl:
		for edge, weight in edge_dct.items():
			src, dst = edge
			line = "%s\t%s\t%d\n" % (src, dst, weight)
			fl.write(line.encode("utf8"))

		# print edge_dct
