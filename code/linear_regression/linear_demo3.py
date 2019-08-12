import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from setting import CHOSE_LANGUAGE
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import datetime
import random

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
coef_list = [0.01, 0.013, 0.019, 0.018, 0.01, 0.013, 0.014, 0.016, 0.016, 0.026, 0.011, 0.029, 0.02, 0.011]
coef_list_ = [1.381, 1.083, 0.747, 0.767, 1.47, 1.053, 1.0, 0.876, 0.888, 0.53, 1.261, 0.481, 0.69, 1.292]
# C 0.01
# C# 0.013
# C++ 0.019
# CODE_FILE 0.018
# CONFIG 0.01
# HTML 0.013
# JAVA 0.014
# JAVASCRIPT 0.016
# PYTHON 0.016
# SCRIPT 0.026
# SQL 0.011
# SWIFT 0.029
# UNIDENTIFIED 0.02
# WEB 0.011

name_list = [
    'C', 'C#', 'C++', 'CODE_FILE', 'CONFIG', 'HTML',
    'JAVA', 'JAVASCRIPT', 'PYTHON', 'SCRIPT', 'SQL',
    'SWIFT', 'UNIDENTIFIED', 'WEB']


class FeatureTest(object):

    def read_csv(self, use_list, file_name='test2.csv',):
        # use_list = [i for i in range(4, 17)]
        df = pd.read_csv(file_name, usecols=use_list,header=None)
        return df

    def count(self, x, y, x_test=None):
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        coef = model.coef_
        score = model.score(x, y)
        # train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)
        return coef, score, test_predict

    def filter_value(self):
        pass

    def get_value(self):
        x = self.read_csv(list(range(15)))
        print(x.head())
        x_list = x.values.tolist()
        print('num is', len(x_list))
        for i in range(14):
            df = x[x[i] > 0]
            if len(df) < 100:
                print('====', len(df))
                values_list = df.values.tolist()
                for i in values_list:
                    if i in x_list:
                        x_list.remove(i)
        print('num is', len(x_list))
        x = DataFrame(x_list)
        x['result'] = x[0]*coef_list[0] + x[1]*coef_list[1] +\
                      x[2]*coef_list[2] + x[3]*coef_list[3] +\
                      x[4]*coef_list[4] + x[5]*coef_list[5] +\
                      x[6]*coef_list[6] + x[7]*coef_list[7] +\
                      x[8]*coef_list[8] + x[9]*coef_list[9] +\
                      x[10]*coef_list[10] + x[11]*coef_list[11] +\
                      x[12]*coef_list[12] + x[13]*coef_list[13]
        x['result'] = x['result'] / x[14]
        print(x['result'].head())
        # x['result2'] = (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13])/x[14]
        boxplot_value = x['result']
        min_value, max_value = self.get_boxplot(boxplot_value)
        # print(x_value, '\n', y_value)
        filter_value = x[x['result'] > min_value]
        print(len(filter_value))
        filter_value = filter_value[filter_value['result'] < max_value]
        print(filter_value.head())
        print(len(filter_value))
        # print(len(x[x['result2'] > 1000]))

        # x_value = DataFrame(x['result'])
        x_value = filter_value[list(range(14))]
        y_value = filter_value[14]
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filter_value.to_csv('filter-{}.csv'.format(now))
        x.to_csv('test-{}.csv'.format(now))
        return x_value, y_value

    def get_boxplot(self, value):
        data = pd.DataFrame({'boxplot':value})
        result = data.boxplot(return_type='dict')
        min_value = result['caps'][0].get_ydata().tolist()[0]
        max_value = result['caps'][1].get_ydata().tolist()[0]
        print('======min', min_value, type(min_value))
        print('======max', max_value)
        # min_value = int(min_value)
        # print('===int min',min_value)
        # print('===int max',max_value)
        # max_value = int(max_value)
        plt.show()
        return min_value,max_value

    def main(self):
        x, y = self.get_value()

        x, x_test, y, y_test = train_test_split(x, y, test_size=0.05)

        # poly = PolynomialFeatures(degree=4)
        # poly.fit(x)
        # x = poly.fit_transform(x)
        # x_test = poly.transform(x_test)
        coef, score, test_predict = self.count(x, y, x_test)
        coef = coef.tolist()
        print(coef)

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
            value_dict.update({key: s})

        java_value = value_dict.get(CHOSE_LANGUAGE)
        print('===== ======')
        # set java value as 1.
        for key, value in value_dict.items():
            if value <= 0.0:
                value_dict.update({key: 0})
                print(key, value)
                continue
            convert_value = java_value / value
            convert_value = round(convert_value, 3)
            print(key, convert_value)
            value_dict.update({key: convert_value})
        print('=======')
        print(score)


f = FeatureTest()
f.main()


