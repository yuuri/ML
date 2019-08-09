import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from setting import CHOSE_LANGUAGE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)

name_list = [
    'C', 'C#', 'C++', 'CODE_FILE', 'CONFIG', 'HTML',
    'JAVA', 'JAVASCRIPT', 'PYTHON', 'SCRIPT', 'SQL',
    'SWIFT', 'UNIDENTIFIED', 'WEB']


class FeatureTest(object):

    def read_csv(self, use_list, file_name='test2.csv',):
        # use_list = [i for i in range(4, 17)]
        df = pd.read_csv(file_name, usecols=use_list)
        return df

    def count(self, x, y, x_test=None):
        model = LinearRegression(normalize=True,fit_intercept=False)
        model.fit(x, y)
        coef = model.coef_
        score = model.score(x, y)
        # train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)
        return coef, score,test_predict

    def main(self):

        x_list = [i for i in range(0, 14)]
        y_list = [14]

        x = self.read_csv(x_list).values
        y = self.read_csv(y_list).values

        x, x_test, y, y_test = train_test_split(x, y, test_size=0.05)

        # poly = PolynomialFeatures(degree=4)
        # poly.fit(x)
        # x = poly.fit_transform(x)
        # x_test = poly.transform(x_test)
        print(type(x))
        coef, score, test_predict = self.count(x, y, x_test)
        coef = coef.tolist()[0]

        # predict result
        # test_predict = test_predict.tolist()
        # y_test = y_test.tolist()
        # print(len(test_predict))
        # print(len(y_test))
        # result = list(zip(test_predict,y_test))
        # for i in result:
        #     print(i)

        value_dict = dict()
        # build dict
        for key, value in zip(name_list, coef):
            # value = value * 100
            s = round(value, 3)

            print(key, s)
            value_dict.update({key: value})

        java_value = value_dict.get(CHOSE_LANGUAGE)
        print('===== ======')
        # set java value as 1.
        for key, value in value_dict.items():
            convert_value = java_value / value
            convert_value = round(convert_value, 3)
            print(key, convert_value)
            value_dict.update({key: convert_value})
        print('=======')
        print(score)


f = FeatureTest()
f.main()


