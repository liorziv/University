import pandas as pd
from Article_Factory import Article_Factory

import sys
import glob
import pickle

SEP = ";"

def save_as_df(article_list, csv_name):
    header = ["pubmed_id", "doi_id", "title", "researchers", "date", "abstract", "journal"]
    df = pd.DataFrame(columns = header)

    for i, article in enumerate(article_list):
        row_lst = []
        row_lst.extend([article.get_pubmed_id(), article.get_doi_id(), article.get_title()])
        researchers = article.get_researchers()
        researchers_string = ""
        if researchers is not None:
            for researcher in researchers:
                researchers_string += researcher.get_full_name() + SEP
        row_lst.append(researchers_string)

        date_tuple  = article.get_date()
        if date_tuple is not None:
            row_lst.append(date_tuple[0] + SEP + date_tuple[1] + SEP + date_tuple[2])
        else:
            row_lst.append("")

        row_lst.append(article.get_abstract())

        row_lst.append(article.get_journal())

        df.loc[i] = row_lst

    print df

    df.to_csv(csv_name+".csv", index = False, sep = "\t", encoding='utf-8')


def main():
    workdir = sys.argv[1]


    for xml_path in glob.glob("%s/*.xml" % workdir):
        article_factory = Article_Factory(xml_path = xml_path)
        article_list = article_factory.make_article_list()

        with open(xml_path[:-3] + "bin", "wb") as fl:
            pickle.dump(article_list, fl)

        # save_as_df(article_list, xml_path[:-4])


if __name__ == "__main__":
    main()
