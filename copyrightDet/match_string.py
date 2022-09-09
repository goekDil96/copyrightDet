from dataclasses import replace
from commonregex import CommonRegex
import re
import spacy
from spacy import displacy
import en_core_web_sm
import string
nlp = spacy.load("en_core_web_sm")


class MatchString:
    def match_email(self, copy_st: list):
        for index in range(len(copy_st)):
            email_matches = CommonRegex(copy_st[index]).emails
            for _ in email_matches:
                copy_st[index] = "DETECTED_EMAIL"
        return copy_st

    def match_url(self, copy_st: str):
        for index in range(len(copy_st)):
            url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
            url_matches = [i[0] for i in re.findall(pattern=url_pattern, string=copy_st[index])]
            for _ in url_matches:
                copy_st[index] = "DETECTED_URL"
        return copy_st

    def match_year(self, copy_st: str):
        years = [str(i) for i in range(1975, 2023)]
        years.extend(["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"])
        years.extend([i for i in range(75, 100)])
        years.extend([i for i in range(10, 23)])

        year_pattern = ""
        for year in years:
            year_pattern += f"\\b{year}\\b|\\b{year}&|^{year}\\b"
            if year != years[-1]:
                year_pattern += "|"

        copy_st = re.sub(pattern=year_pattern, string=copy_st, repl=" DETECTED_YEAR ", count=2)
        return copy_st
    
    def replace_copyright(self, copy_st : str) -> str:
        copy_st = copy_st.lower()
        copy_st = copy_st.replace("\u00a9" , "(c)")
        len_1 = len(copy_st)
        copy_st = re.sub(pattern=r"\bcopyright\b|\bcopyright&|^copyright\b|\(\bc\b\)|\(\bc\)&|^\(c\b\)|&\bcopy\b|&\bcopy&|^&copy\b", repl="detected_copyright", string=copy_st)
        len_2 = len(copy_st)
        # just one space between words
        copy_st = re.sub("\s{2,}", " " , copy_st)
        if len_1 - len_2 == 0:
            returnValue = 1
        else:
            returnValue = 0
        return copy_st, returnValue

    def match_corp_and_person(self, copy_st : str) -> str:
        doc = nlp(copy_st)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON"] and "portions" not in ent.text.lower():
                copy_st = copy_st.replace(ent.text, f" DETECTED_{ent.label_} ")
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        return copy_st
        
    def replace_punctuation(self, copy_st: str) -> str:
        # get rid of punctuation
        copy_st = re.sub("\[.*?\]", " ", copy_st)
        copy_st = re.sub("[%s]" % re.escape(string.punctuation.replace("_", "")), " ", copy_st)

        # just one space between words
        copy_st = re.sub("\s{2,}", " " , copy_st)
        return copy_st

    def preprocess(self, copy_st: str) -> bool:
        # just one space between words
        # copy_st = re.sub("\s{2,}", " " , copy_st)
        copy_st = self.match_year(copy_st)
        copy_st = self.match_corp_and_person(copy_st)
        copy_st = self.replace_copyright(copy_st)[0]
        copy_st = copy_st.split()
        copy_st = self.match_email(copy_st)
        copy_st = self.match_url(copy_st)
        copy_st = [term for term in copy_st if not term.isdigit()]
        if copy_st.count("detected_copyright") >= 2 and len(copy_st) >= 2:
            index_1 = copy_st.index("detected_copyright")
            index_2 = copy_st.index("detected_copyright", index_1 + 1)
            if index_2 - index_1 > 1:
                copy_st.append("word_between_copyright")
        copy_st = " ".join(copy_st)
        copy_st = self.replace_punctuation(copy_st)
        copy_st = copy_st.split()
        test_list1 = list(filter(lambda x: "detected_" not in x.lower(), copy_st))
        if len(test_list1) > 0:
            copy_st.append("detected_other_words")
        copy_st = " ".join(copy_st)
        copy_st = copy_st.lower()
        return copy_st
        


def main():
    # doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)

    corpus_url = ["Copyright (C) 2005-2007  Kristian Hoegsberg <krh@bitplanet.net>"]

    match_string = MatchString()
    for i in corpus_url:
        print(match_string.preprocess(i)
                # , CommonRegex(i).emails
                )


if __name__ == "__main__":
    main()

