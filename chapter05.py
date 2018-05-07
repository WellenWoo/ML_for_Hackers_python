import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

matplotlib.style.use('ggplot')

f0 = r'data\longevity.csv'
f1 = r'data\01_heights_weights_genders.csv'
f2 = r'data\top_1000_sites.tsv'

def plot_age_smoke(fn):
    '''绘制吸烟/不吸烟-年龄 直方图'''
    d0 = pd.read_csv(fn)
    d1 = d0[d0.Smokes<1]
    d2 = d0[d0.Smokes>0]

    ##fig,(ax1,ax2) = plt.subplots(2,1,sharex = True)
    ax1 = sns.distplot(d1.AgeAtDeath,hist = False,color = 'g',kde_kws = {'shade':True,
                                                                     'label':'No Smoke'})
    ax2 = sns.distplot(d2.AgeAtDeath,hist = False,color = 'r',kde_kws = {'shade':True,
                                                                     'label':'Smoke'})
    
    ##ax1 = d1.AgeAtDeath.plot.hist(bins = 5)
    ##ax2 = d2.AgeAtDeath.plot.hist(bins = 5)

    plt.tight_layout()
    plt.show()

def plot_weight_height(fn):
    '''绘制身高-体重散点图'''
    d0 = pd.read_csv(fn)
    hdata = np.array(d0['Height'])
    wdata = np.array(d0['Weight'])
    plt.scatter(hdata,wdata)
    plt.title('height vs weight')
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.show()
    

def linearReg(fn):
    '''构建身高-体重的线性模型'''
    d0 = pd.read_csv(fn)
    hdata = np.array(d0['Height'])
    wdata = np.array(d0['Weight'])
    X_train,X_test,y_train,y_test = train_test_split(hdata,wdata)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    model = LinearRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    coef = model.coef_
    intercept = model.intercept_
    R_square = model.score(X_test,y_test)

    print (u'coef:{coef};intercept:{intercept};R方:{R_square}'.format(coef=coef,
                                                                     intercept=intercept,
                                                                     R_square=R_square))
##    predicted.weight == -350.446944876 + 7.71223775 * observed.height
    plt.scatter(hdata,wdata)
    func = intercept+coef*hdata
    plt.plot(hdata,func,lw = 1,color = 'r')
    plt.show()
    return model

def plot_pageviews(fn):
    '''绘制 Pageviews-UniqueVisitors散点图'''
    d0 = pd.read_table(fn)

    d0.plot.scatter(x = 'PageViews',y = 'UniqueVisitors')
    d3.plot.density(x = 'PageViews',y = 'UniqueVisitors',logx = True)
    d0.plot.scatter(x = 'PageViews',y = 'UniqueVisitors',logx = True,logy = True)
    plt.show()

def loglinearReg(fn):
    '''构建log(Pageviews)-log(UniqueVisitors)回归模型'''
    d0 = pd.read_table(fn)
    X = np.log(d0['PageViews'])
    y = np.log(d0['UniqueVisitors'])
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)

    model = LinearRegression()
    model.fit(X_train,y_train)

    coef = model.coef_
    intercept = model.intercept_
    R_square = model.score(X_test,y_test)

    print (u'coef:{coef};intercept:{intercept};R方:{R_square}'.format(coef=coef,
                                                                     intercept=intercept,
                                                                     R_square=R_square))
    d0.plot.scatter(x = 'PageViews',y = 'UniqueVisitors',logx = True,logy = True)
    func = np.exp(intercept+coef*np.log(d0['PageViews']))
    plt.plot(d0['PageViews'],func,lw = 1,color = 'r')
    plt.show()
    return model  
