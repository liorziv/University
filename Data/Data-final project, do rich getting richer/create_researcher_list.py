from Researcher import Researcher
from Article import Article
import pickle
import sys
import glob

def generate_researcher_list(researcher_dct, article_list):

	for article in article_list:
		for author in article.get_researchers():

			if author not in researcher_dct:
				researcher_dct[author.get_full_name()] = author
				
			researcher_dct[author.get_full_name()].add_article(article)


if __name__ == "__main__":

	workdir = sys.argv[1]

	researcher_dct = {}

	for article_file_path in glob.glob("%s/*.bin" % workdir)[:40]:

		with open (article_file_path, "rb") as fl:
			article_list = pickle.load(fl)

		generate_researcher_list(researcher_dct, article_list)

	with open("researchers.bin", "wb") as fl:
		pickle.dump(researcher_dct, fl)

			

			
	