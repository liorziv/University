import xml.etree.ElementTree as ET
from Article import Article
from Researcher import Researcher
import copy

class Article_Factory(object):

    def __init__(self, xml_path):
        self.__xml_path = xml_path
        self.__tree = None
        self.__root = None
        self.__article_lst = []


    def make_article_list(self):
        self.__tree = ET.parse(self.__xml_path)
        self.__root = self.__tree.getroot()

        # iterate over articles in xml
        for article in self.__root.iter(tag = "PubmedArticle"):

            # initialize empty containers. this is important because of the "try-except" thingy
            pubmed_id = None
            doi_id = None
            title = None
            researchers = []
            date = None
            abstract = None
            journal = None

            try:
                medline_element = article.find("MedlineCitation")

                try:
                    article_element =  medline_element.find("Article")

                    # get title
                    try:
                        title = article_element.find("ArticleTitle").text
                    except AttributeError:
                        print "no title"

                    #  get reasearchers names
                    try:
                        researchers_elements = article_element.find("AuthorList").findall("Author")
                        for researcher_element in researchers_elements:
                            last_name = researcher_element.find("LastName").text
                            fore_name = researcher_element.find("ForeName").text
                            initials = researcher_element.find("Initials").text
                            affiliation = researcher_element.find("AffiliationInfo").text

                            researchers.append(Researcher(fore_name, last_name, initials, affiliation))
                    except AttributeError:
                        print "no reasearchers names"

                    # get date
                    try:
                        date_element = article_element.find("ArticleDate")
                        date = (date_element.find("Day").text,
                                date_element.find("Month").text,
                                date_element.find("Year").text)
                    except AttributeError:
                        print "no date"

                    # get abstract
                    try:
                        abstract= ""
                        abstract_elements = article_element.find("Abstract").findall("AbstractText")
                        for sub_abstract in abstract_elements:
                            if sub_abstract.text is not None:
                                abstract += sub_abstract.text + " "

                        abstract = abstract.replace("\n", " ")

                    except AttributeError:
                        print "no abstract"

                except AttributeError:
                    print "no article"


                # get journal name
                try:
                    journal = medline_element.find("MedlineJournalInfo").find("MedlineTA").text

                except AttributeError:
                    print "no journal"
            except AttributeError:
                print "no MedlineCitation"

            # get article ids
            try:
                id_element_lst = article.find("PubmedData").find("ArticleIdList").findall("ArticleId")
                for id_element in id_element_lst:
                    try:
                        if id_element.attrib["IdType"] == "pubmed":
                            pubmed_id = id_element.text
                        elif id_element.attrib["IdType"] == "doi":
                            doi_id = id_element.text
                    except KeyError:
                        print "missing one article id: pubmed or doi"

            except AttributeError:
                print "no article id"

            new_article = Article(pubmed_id = pubmed_id,
                                  doi_id = doi_id,
                                  title = title,
                                  researchers = researchers,
                                  date = date,
                                  abstract = abstract,
                                  journal = journal)
            self.__article_lst.append(new_article)

            # new_article.print_article()

        return copy.copy(self.__article_lst)


import sys
def main():
    xml_path = sys.argv[1]
    article_factory = Article_Factory(xml_path = xml_path)
    article_factory.make_article_list()


if __name__ == "__main__":
    main()
