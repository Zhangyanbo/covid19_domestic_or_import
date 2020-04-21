import csv
import json
import linecache
import datetime
import sys
import numpy as np
import random
from scipy.sparse import *
from scipy import *
import scipy as sp
import matplotlib.pyplot as plt
import os.path
import matplotlib
import copy
import pandas as pd
from scipy import stats
from tempfile import TemporaryFile
import pickle
from scipy.integrate import odeint
from torchdiffeq import odeint as dodeint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import country_converter as coco

def flushPrint(variable):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % variable)
    sys.stdout.flush()

class COVID19:
    def __init__(self, flowdata='data/country_flow/', casedata='data/global0415_en.csv', populationdata='data/country_population.csv'):
        # 首先，我们要校准各个国家的名称，做法是分别把航空流量数据、国家人口数据和病例数据中的国家字段都加载进来，统一转换为标准国家名称
        # flow data
        self._name_combine(flowdata, casedata, populationdata)
        self._read_pop(populationdata)
        self._read_flow_dynamics(flowdata)
        self._read_case_data(casedata)
        
    def _name_combine(self, flowdata, casedata, populationdata):
        # 首先，我们要校准各个国家的名称，做法是分别把航空流量数据、国家人口数据和病例数据中的国家字段都加载进来，统一转换为标准国家名称
        # flow data
        first_date = datetime.datetime(2019, 12, 1, 0, 0)
        day_len = (datetime.datetime.strptime('2020-12-31','%Y-%m-%d') - first_date).days
        fijt = []
        countries = set([])
        for day in range(day_len):
            dd=(first_date+datetime.timedelta(days=day)).strftime("%Y-%m-%d")
            flushPrint(dd)
            filename = flowdata+dd+'.csv'
            df=pd.read_csv(filename)
            countries1=set(df['Unnamed: 0'])
            countries2=set(list(df.columns)[1:])
            countries = countries | countries1 | countries2
            flushPrint(len(countries))
        countries_temp = list(coco.convert(names=countries, to="name_short"))
        countries1 = set(countries_temp)

        self.name_map = {}
        for i,cc in enumerate(countries):
            vv = self.name_map.get(cc, '')
            if len(vv) == 0:
                self.name_map[cc] = countries_temp[i]
        print(len(countries), len(countries1))

        # case data
        df = pd.read_csv(casedata)
        countries = set(df['country'][1:])
        countries2 = set(coco.convert(names=countries, to="name_short"))
        print(len(countries), len(countries2))

        # population data
        df = pd.read_csv(populationdata)
        countries=set(df['Name'][1:])
        countries3 = set(coco.convert(names=countries, to="name_short"))
        print(len(countries), len(countries3))

        countries = countries1 | countries2 | countries3
        self.nodes = {}
        for country in countries:
            if country != 'not found':
                idx = self.nodes.get(country, len(self.nodes))
                self.nodes[country] = idx

        #这一步的输出结果主要是一个字典nodes，国家名称作为key，国家在nodes中的位置作为value，这个值也是后面各个数组的国家索引
        #另外还输出一个name_map，它的作用是把不标准的国家名称映射为标准名称
    def _read_pop(self, populationdata):
        #开始正是读取人口数据
        df = pd.read_csv(populationdata)
        country_properties1 = {}
        for index, row in df.iterrows():
            try:
                ccc = coco.convert(names=row['Name'], to="name_short")
                country_properties1[ccc] = float(row['Population'])
            except:
                print(row['Name'])
                print(row['Population'])

        self.country_pop = {}
        print('second')
        for country, i in self.nodes.items():
            if self.country_pop.get(country, -1)>0:
                print('duplicate:', country)
            else:
                pop = country_properties1.get(country, 1)
                self.country_pop[country] = pop
        print('These countries have no pop data:')
        for k,v in self.country_pop.items():
            if v <= 0:
                print(k)
        self.population = np.ones(len(self.nodes))
        for cc,i in self.nodes.items():
            self.population[i] = self.country_pop.get(cc, 1.0)
        #这部分输出是数组population，国家的顺序按照nodes中记载的顺序来，以及一个字典country_pop，key为国家名，value为人口数
    def _read_flow_dynamics(self, flowdata):
        # Read inter-country flow data day by day:
        #读取航空流量数据，从2019年12月1日到2020年12月31日
        first_date = datetime.datetime(2019, 12, 1, 0, 0)
        day_len = (datetime.datetime.strptime('2020-12-31','%Y-%m-%d') - first_date).days
        self.fijt = []
        for day in range(day_len):
            dd=(first_date+datetime.timedelta(days=day)).strftime("%Y-%m-%d")
            flushPrint(dd)
            filename = flowdata+dd+'.csv'
            df=pd.read_csv(filename)
            fij = np.zeros([len(self.nodes), len(self.nodes)])
            for index, row in df.iterrows():

                for i,content in enumerate(row.iteritems()):
                    if i==0:
                        country_i = self.name_map[content[1]]
                    else:
                        country_j = self.name_map[content[0]]
                        if country_j!='not found' and country_i!='not found':
                            if self.country_pop[country_i] > 0:
                                fij[self.nodes[country_i], self.nodes[country_j]] += float(content[1])
                            else:
                                fij[self.nodes[country_i], self.nodes[country_j]] += 0#float(content[1])
                            if self.nodes[country_i]==self.nodes[country_j]:
                                fij[self.nodes[country_i], self.nodes[country_j]] = 0
            for k,pop in self.country_pop.items():
                flux1 = fij[self.nodes[k],:].sum()
                flux2 = fij[:, self.nodes[k]].sum()
                flux =max(flux1, flux2)
                if flux > self.country_pop[k]:
                    self.country_pop[k] = flux
                    self.population[self.nodes[k]] = flux
                fij[self.nodes[k],:] = fij[self.nodes[k],:] / self.country_pop[k]

            self.fijt.append(fij)

        #输出为一个列表fijt，记录每一天的流量矩阵。流量矩阵fij表示从i到j的流占国家i总人口的比例
    def _read_case_data(self, casedata):
        # 读取病例数据，从2019年12月1日，到2020年4月15日
        df = pd.read_csv(casedata)

        all_cases_countries = list(set(df['country']))[1:]

        china=df.loc[df['country']=='China',['cum_confirm','time','cum_heal','cum_dead']]
        dates = list(china['time'])
        sorted_dates = np.sort(dates)
        self.first_date = datetime.datetime(2020, 1, 1, 0, 0)
        self.start_date = datetime.datetime(2019, 12, 1, 0, 0)
        day_len = (datetime.datetime.strptime('2020/4/16','%Y/%m/%d') - self.start_date).days
        self.first_cases = int(china.loc[china['time']=='2019/12/1']['cum_confirm'])

        all_confirmed_cases = np.zeros([day_len, len(self.nodes)])
        all_cued_cases = np.zeros([day_len, len(self.nodes)])
        all_death_cases = np.zeros([day_len, len(self.nodes)])
        for country in all_cases_countries:
            #country = country_map2.get(country, country)
            ccc = coco.convert(names=country, to="name_short")
            if self.nodes.get(ccc, -1)>=0:
                subset = df.loc[df['country']==country,['cum_confirm','time','cum_heal','cum_dead']]
                if len(subset)>0:
                    new_cases = np.array(subset['cum_confirm'])
                    cued_cases = np.array(subset['cum_heal'])
                    die_cases = np.array(subset['cum_dead'])


                    dates = list(subset['time'])
                    new_cases = np.r_[new_cases[0],np.diff(new_cases)]
                    cued_cases = np.r_[cued_cases[0],np.diff(cued_cases)]
                    die_cases = np.r_[die_cases[0],np.diff(die_cases)]
                    for i,dd in enumerate(dates):
                        if pd.isnull(dd):
                            dd = dates1[i]
                        if not pd.isnull(dd):

                            day=(datetime.datetime.strptime(dd,'%Y/%m/%d') - self.start_date).days
                            all_confirmed_cases[day, self.nodes[ccc]] += new_cases[i]
                            all_cued_cases[day, self.nodes[ccc]] += cued_cases[i]
                            all_death_cases[day, self.nodes[ccc]] += die_cases[i]

        time_cases = np.arange(day_len) - (self.first_date - self.start_date).days
        self.all_cumconfirm_cases = np.cumsum(all_confirmed_cases, 0) - np.cumsum(all_cued_cases, 0) - np.cumsum(all_death_cases, 0)
        self.all_cumexist_cases = np.cumsum(all_confirmed_cases, 0)
        for i in range(self.all_cumconfirm_cases.shape[1]):
            yy = self.all_cumconfirm_cases[:, i]
            bools = yy>0
            plt.semilogy(time_cases[bools], yy[bools],'.')
        plt.show()


        self.targets = torch.cat((torch.Tensor(self.all_cumconfirm_cases), torch.Tensor(self.all_cumexist_cases)),1)
        for i in range(len(self.population)):
            self.targets[:, i] /= self.population[i]
            self.targets[:, len(self.nodes)+i]/=self.population[i]
            if self.population[i]==1:
                self.targets[:,i] = 0
                self.targets[:,i+len(self.nodes)]=1
        self.mask = self.targets > 0

        #这部分输出有如下一些数据：
        #1、first_date：2020年1月1日，是我们的时间0点
        #2、start_date: 2019年12月1日，是病例数据起始的0点
        #3、all_cumconfirm_cases: 是一个time_length*国家数的而为数组，记录了某一天现存的确诊病例数，这里time_length就是12-1日到4-15日的时间
        #4、all_cumexist_cases：是一个time_length*国家数的而为数组，记录了某一天累积的确诊病例数
        #5、targets，是一个Tensor，time_length*2*国家数维度，将all_cumconfirm_cases和all_cumexist_cases合并到了一起，作为训练的target
        #6、mask：是一个Tensor，time_length*2*国家数维度，0-1矩阵，记录了targets是否>0的情况
        #7、first_cases: 在2019年12月1日，中国的确诊病例数