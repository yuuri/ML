from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.externals import joblib

MODEL_NAME = 'Linear.m'


class Linear(object):
    def __init__(self):
        pass
    
    def create_model(self,x,y):
        model = LinearRegression()
        model.fit(x,y)
        coef = model.coef_
        intercept = model.intercept_
        score = model.score(x, y)
        self.save_model(model)
        return coef,intercept,score
    
    def get_data(self):
        x, y = datasets.make_regression(n_samples=300, n_features=3, n_targets=1, noise=5)
        # plt.figure()
        # plt.scatter(x,y)
        # plt.show()
        return x, y
    
    def set_train_split(self):
        pass
    
    def pre_procession(self,data):
        return preprocessing.scale(data)
    
    def save_model(self,model):
        joblib.dump(model,MODEL_NAME)
    
    def restore_model(self):
        return joblib.load(MODEL_NAME)
    
    def main(self):
        x,y = self.get_data()
        # x = self.pre_procession(x)
        # y = self.pre_procession(y)
        coef, intercept, score = self.create_model(x, y)
        print('==== coef ====')
        print(coef, )
        print('==== intercept ====')
        print(intercept)
        print('==== score ====')
        print(score)


run = Linear()
run.main()
# ==== coef ====
# [0.39072646 0.29663471 0.54071077 0.09709144 0.40517252 0.3021027
#  0.21999778 0.05294022 0.21614554 0.42508272]
# ==== intercept ====
# 9.965950341549611e-17
# ==== score ====
# 0.9992915426769156

