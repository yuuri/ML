import pandas as pd
from test.feature_code import FeatureTest
from setting import CHOSE_LANGUAGE
from pandas.core.frame import DataFrame
from test.feature_code import name_list, coef_list

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)

app = FeatureTest()

coef_dict = dict(zip(name_list, coef_list))
print(coef_dict)

content = pd.read_csv('test3.csv')

# 获取代码行数据
result = content.iloc[::1, 4:-1]

# 获取工时
y_value = content.iloc[::1, -1]
print(y_value.head())
y_value_name = y_value.name
# 批量修改列名
result.rename(columns=lambda x: x.split('_')[1], inplace=True)
column_list = result.columns.tolist()
column_list.append(y_value_name)
result[y_value_name] = y_value

# 根据有效代码条数筛选数据
value_list = app.filter_value(result, column_list, filter_num=100)
value = DataFrame(value_list)

# 给新创建的data frame 添加columns信息
value.columns = column_list

# 由上一次系数求生产率
value['result'] = 0
for index, row in value.iteritems():
    if index in coef_dict:
        count = row * coef_dict[index]
        value['result'] += count

print(value.head())
# value['result'] = value['result'] / value.iloc[::1, :2]
value['result'] = value['result'] / value[y_value_name]
# 画 boxplot
boxplot_value = value['result']
print('start is', len(value))
min_value, max_value, means, medians = app.get_boxplot(boxplot_value)

# 筛选数据
if min_value <= 0:
    min_value = 0.0
filter_value = value[value['result'] > min_value]
filter_value = filter_value[filter_value['result'] < max_value]

x_value = filter_value.iloc[::1, :-2]
y_value = filter_value.iloc[::1, -2]
print('filter is ', len(filter_value))


# 求系数
coef,score = app.count(x_value, y_value)
print(x_value.columns)
coef = coef.tolist()
print(coef)

name_list2 = filter_value.columns.tolist()[:-1]
print(filter_value.columns.tolist())
# 将系数合并为一个字典
value_dict = dict()
for key, value in zip(name_list2, coef):
    # value = value * 100
    s = round(value, 5)

    print(key,s)
    value_dict.update({key: s})

# 求转换系数
format_value = value_dict.get(CHOSE_LANGUAGE)
print('===== ======')
# set java value as 1.
convert_value_dict = dict()
for key,value in value_dict.items():
    if value <= 0.0:
        convert_value_dict.update({key: 0})
        print(key, value)
        continue
    convert_value = value / format_value
    convert_value = round(convert_value,3)
    print(key,convert_value)
    convert_value_dict.update({key: convert_value})

# 求生产率
filter_value['result'] = 0
for index, row in filter_value.iloc[::1, :-2].iteritems():
    count = row * convert_value_dict[index]
    filter_value['result'] += count

filter_value['result'] = filter_value['result'] / filter_value[y_value_name]
min_value, max_value, means, medians = app.get_boxplot(filter_value['result'])

print('===== 生产率 ====')
print('平均数为 {}'.format(means))
print('中位数为 {}'.format(medians))
print('最大值为 {}'.format(max_value))
print('最小值为 {}'.format(min_value))



