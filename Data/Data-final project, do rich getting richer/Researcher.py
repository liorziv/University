import copy

class Researcher(object):
    def __init__(self, fore_name, last_name, initials, affiliation):
        """
        :param fore_name: string
        :param last_name: string
        :param initials: string
        :param affiliation: string
        """
        self.__fore_name = fore_name
        self.__last_name = last_name
        self.__initials = initials
        self.__affiliation = affiliation

        self.__article_list = []
        self.__institue_list = []


    def add_article(self, article):
        if article not in self.__article_list:
            self.__article_list.append(article)


    def get_articles(self):
        return copy.copy(self.__article_list)


    def get_institutes(self):
        return copy.copy(self.__article_list)


    def get_full_name(self):
        return self.__fore_name + " " + self.__last_name


    def get_affiliation(self):
        return self.__affiliation
