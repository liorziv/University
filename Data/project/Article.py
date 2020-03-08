import copy
import Abstract

class Article(object):

    def __init__(self, pubmed_id, doi_id, title, researchers, date, abstract, journal):
        """
        for all objects that are not a list, the empty state is None. for list, the empty state is [].
        :param pubmed_id: string
        :param doi_id: string
        :param title: string
        :param researchers: list of researchers
        :param date: tuple of (int day, int month, int year)
        :param abstract: string
        :param journal: string
        """

        self.__pubmed_id = pubmed_id
        self.__doi_id = doi_id
        self.__title = title
        self.__researchers = researchers
        self.__date = date
        self.__abstract = abstract
        self.__journal = journal


    def get_pubmed_id(self):
        return copy.copy(self.__pubmed_id)


    def get_doi_id(self):
        return copy.copy(self.__doi_id)

    def get_title(self):
        return copy.copy(self.__title)

    def get_researchers(self):
        return copy.copy(self.__researchers)

    def get_date(self):
        return copy.copy(self.__date)

    def get_abstract(self):
        return copy.copy(self.__abstract)

    def get_journal(self):
        return copy.copy(self.__journal)

    def print_article(self):
        print "Article:"
        print "pubmed_id:"
        print self.__pubmed_id
        print "doi_id:"
        print  self.__doi_id
        print "title:"
        print  self.__title
        try:
            print "researchers:"
            print  [researcher.get_full_name() for researcher in self.__researchers]

        except AttributeError:
            pass
        print "date:"
        print  self.__date
        print "abstract:"
        print self.__abstract
        print "journal", self.__journal
        print "=" * 60
