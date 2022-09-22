from .match_string import MatchString
import os
import json


class RuleBased:
    dict_whole_rule = {
        # strings without further words (other than copyright and year)
        "detected_copyright": 1.0,
        "detected_copyright detected_copyright": 1.0,
        "detected_copyright detected_year": 1.0,
        "detected_copyright detected_copyright detected_year": 1.0,
        "detected_copyright detected_copyright detected_year detected_year": 1.0,
        "detected_copyright detected_person": 0.0,
        "detected_copyright detected_org": 0.0,
        "detected_copyright detected_email": 0.0,
        "detected_copyright detected_url": 0.0,
        "detected_copyright detected_person detected_email": 0.0
    }

    dict_partial_rule = {
        "all rights reserved": 0.0,
        "portions detected_copyright": 0.0,
        # strings with clear classes person
        "detected_copyright detected_copyright detected_person": 0.0,
        "detected_copyright detected_copyright detected_year detected_person": 0.0,
        "detected_copyright detected_copyright detected_person detected_year": 0.0,
        "detected_copyright detected_copyright detected_year detected_year detected_person": 0.0,
        # strings with clear classes org
        "detected_copyright detected_copyright detected_org": 0.0,
        "detected_copyright detected_copyright detected_year detected_org": 0.0,
        "detected_copyright detected_copyright detected_org detected_year": 0.0,
        "detected_copyright detected_copyright detected_year detected_year detected_org": 0.0,
        # strings with clear classes email
        # "detected_copyright the": 0.0,
        "detected_copyright detected_copyright detected_email": 0.0,
        "detected_copyright detected_copyright detected_year detected_email": 0.0,
        "detected_copyright detected_copyright detected_email detected_year": 0.0,
        "detected_copyright detected_copyright detected_year detected_year detected_email": 0.0,
        # strings with clear classes url
        "detected_copyright detected_copyright detected_url": 0.0,
        "detected_copyright detected_copyright detected_year detected_url": 0.0,
        "detected_copyright detected_copyright detected_url  detected_year": 0.0,
        "detected_copyright detected_copyright detected_year detected_year detected_url": 0.0,
        # strings with clear classes email and person
        "detected_copyright detected_copyright detected_person detected_email": 0.0,
        "detected_copyright detected_copyright detected_year detected_person detected_email ": 0.0,
        "detected_copyright detected_copyright detected_person detected_email detected_year ": 0.0,
        "detected_copyright detected_copyright detected_year detected_year detected_person detected_email ": 0.0,
    }

    dict_not_in = {
        "detected_copyright": 1.0
    }

    def decide(self, copy_st):
        prePro = MatchString()
        copy_st = prePro.preprocess(copy_st)
        for i in self.dict_not_in.keys():
            if i not in copy_st:
                return self.dict_not_in[i]
        for i in self.dict_partial_rule.keys():
            if i in copy_st:
                return self.dict_partial_rule[i]
        for i in self.dict_whole_rule.keys():
            if copy_st == i:
                return self.dict_whole_rule[i]
        return 0.5

    def predict(self, y):
        predict_y = []
        for i in y:
            predict_y.append(self.decide(i))
        return predict_y


def main():
    with open(os.path.join(os.getcwd(), "data", "pos_neg_copy_y_train.json"), "r", encoding="utf8") as file:
        data = json.load(file)


    corpus = ["copyright The OpenTelemetry Authors"]

    rule_based = RuleBased()
    result = rule_based.predict(corpus)

    print(result.count(1))
    print(result.count(0.5))
    print(result.count(0))
        
if __name__ == "__main__":
    main()

