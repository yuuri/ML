import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression

from setting import CHOSE_LANGUAGE, coef_list, name_list
from common.log import LogAdapter

LOG = LogAdapter().set_log(__name__)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)

SUM_COLUMN_NAME = 'SUM_RESULT'


class FeatureCount(object):
    def __init__(self):
        self.coef_dict = dict(zip(name_list, coef_list))
        self.sum_column_name = SUM_COLUMN_NAME
        self.filter_language = []

    def read_csv(self, file_name):
        """

        :param file_name: csv file name
        :return: DataFrame structure data ,read form csv file.
        """
        # read csv file
        content = pd.read_csv(file_name)
        codeline_value = content.iloc[::1, 4:-1]

        work_time = content.iloc[::1, -1]
        work_time_name = work_time.name

        # change columns name
        value = codeline_value
        value.rename(columns=lambda x: x.split('_')[1], inplace=True)

        # add work time info to value
        column_list = value.columns.tolist()
        column_list.append(work_time_name)
        value[work_time_name] = work_time

        return value

    def filter_value(self, content, language_list=name_list, filter_num=100):
        content_list = content.values.tolist()
        content_columns = content.columns.tolist()
        list_num = len(content_list[0]) - 1
        for i in range(list_num):
            df = content[content.iloc[::1, i] > 0]
            len_num = len(df)
            # print('{} 有效数量为{} '.format(name_list[i],len_num))
            # if someone language data count is less than 100, the data will be remove.
            if len_num < filter_num:
                print('==== {} 语言有效代码不足 100,有效数量为{}'.format(language_list[i], len_num))
                print('==== {} 语言相关数据将不会参与计算'.format(language_list[i]))
                LOG.info('==== {} 语言有效代码不足 100,有效数量为{}'.format(language_list[i], len_num))
                LOG.info('==== {} 语言相关数据将不会参与计算'.format(language_list[i]))
                self.filter_language.append(language_list[i])
                values_list = df.values.tolist()
                for value in values_list:
                    if value in content_list:
                        content_list.remove(value)

        print('num is', len(content_list))
        LOG.info('After Filter num is {}'.format(len(content_list)))
        # restore structure as DataFrame
        content = DataFrame(content_list)
        content.columns = content_columns
        return content

    def count_code_sum(self, filter_value, sum_name):
        filter_value[sum_name] = 0

        for index, row in filter_value.iteritems():
            if index in self.coef_dict:
                count = row * self.coef_dict[index]
                filter_value[sum_name] += count
        work_time_name = filter_value.columns.tolist()[-2]
        filter_value[sum_name] = filter_value[sum_name] / filter_value[work_time_name]
        return filter_value

    def draw_box_plot(self, draw_value, filter_flag=True, show_graph=False):

        data = pd.DataFrame({'box_plot': draw_value})
        result = data.boxplot(return_type='dict', showmeans=True)
        min_value = result['caps'][0].get_ydata().tolist()[0]
        max_value = result['caps'][1].get_ydata().tolist()[0]
        means = result['means'][0].get_ydata().tolist()[0]
        medians = result['medians'][0].get_ydata().tolist()[0]

        if show_graph:
            plt.show()

        if filter_flag:
            return min_value, max_value

        return min_value, max_value, means, medians

    def count_coefficient(self, x, y):
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        coef = model.coef_.tolist()
        score = model.score(x, y)

        return coef, score

    def filter_max_min(self, filter_value, min_value, max_value):
        if min_value <= 0:
            min_value = 0.0

        filter_value = filter_value[filter_value[self.sum_column_name] > min_value]
        filter_value = filter_value[filter_value[self.sum_column_name] < max_value]

        x_value = filter_value.iloc[::1, :-2]
        y_value = filter_value.iloc[::1, -2]

        return x_value, y_value

    def convert_coef(self, language_list, coef):
        # 求转换系数
        value_dict = dict()
        for key, value in zip(language_list, coef):
            # value = value * 100
            if key in self.filter_language:
                continue

            s = round(value, 3)

            print(key, value)
            value_dict.update({key: value})

        format_value = value_dict.get(CHOSE_LANGUAGE)
        # set java value as 1.
        print('\n')

        convert_value_dict = dict()
        for key, value in value_dict.items():
            # 使用绝对值判断是否小于 0
            # value = abs(value)
            if value <= 0.0:
                convert_value_dict.update({key: 0.0})
                print(key, 0.0)
                continue

            convert_value = value / format_value
            convert_value = round(convert_value, 3)
            print(key, convert_value)
            convert_value_dict.update({key: convert_value})

        return convert_value_dict

    def count_productivity(self, x_value, y_value, convert_value_dict):

        x_value[self.sum_column_name] = 0

        for index, row in x_value.iloc[::1, :-1].iteritems():
            if index in self.filter_language:
                continue
            count = row * convert_value_dict[index]
            x_value[self.sum_column_name] += count

        x_value[self.sum_column_name] = x_value[self.sum_column_name] / y_value
        box_value = x_value[self.sum_column_name]
        min_value, max_value, means, medians = self.draw_box_plot(box_value, filter_flag=False, show_graph=True)

        return min_value, max_value, means, medians

    def main(self, file_name='test3.csv'):

        # get data form file
        pd_value = self.read_csv(file_name)

        # get language name list
        language_list = pd_value.columns.tolist()[:-1]

        # filter value
        filter_value = self.filter_value(pd_value, language_list=language_list)

        # count code line sum (use previous time language coefficient)
        sum_value = self.count_code_sum(filter_value, self.sum_column_name)

        # get max_value,min_value by draw boxplot
        min_value, max_value = self.draw_box_plot(sum_value[self.sum_column_name], show_graph=True)

        # if data not in max and min will delete.
        x_value, y_value = self.filter_max_min(sum_value, min_value, max_value)

        # count coefficient
        coef, score = self.count_coefficient(x_value, y_value)

        # convert coef
        convert_value_dict = self.convert_coef(language_list, coef)

        # count productivity
        min_value, max_value, means, medians = self.count_productivity(x_value, y_value, convert_value_dict)

        print('===== 生产率 ====')
        print('平均数为 {}'.format(means))
        print('中位数为 {}'.format(medians))
        print('最大值为 {}'.format(max_value))
        print('最小值为 {}'.format(min_value))

        # import psutil
        # import os
        # process = psutil.Process(os.getpid())
        # print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')

        for key, value in convert_value_dict.items():
            if value == 0:
                continue

            with open('计算结果.txt', 'a') as f:
                f.write('===== {} 生产率 ====\n'.format(key))
                f.write('平均数为 {}\n'.format(means / value))
                f.write('中位数为 {}\n'.format(medians / value))
                f.write('最大值为 {}\n'.format(max_value / value))
                f.write('最小值为 {}\n'.format(min_value / value))
                f.write('\n')


app = FeatureCount()
app.main()
