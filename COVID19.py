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
import gc

def flushPrint(variable):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % variable)
    sys.stdout.flush()
    
# 上述微分方程模拟器，分成了两个，一个是多节点有网络的，底下的是单一节点的
def intervention(parameters,t):
    epsilon = parameters['epsilon']
    lambd = parameters['lambds']
    t0 = parameters['t0s']
    tstar = parameters['tstar']
    lambd2 = parameters['lambds2']
    exp1 = np.exp(lambd * (t - t0) - np.log(1/epsilon-1))
    decay = 1/(1+exp1)
    if tstar>0:
        #tstar = x0 + np.log((1-relax*epsilon)/(relax*epsilon+eta-1))
        exp2 = np.exp(lambd * (2*tstar - t - t0) - np.log(1/epsilon-1))
        decay1 += 1/(1+exp2)
    return decay
def np_ode(states, t, parameters, fijt, population):
    #print(t, states[0])
    sz = states.shape[0] // 5
    us = states[:sz]
    cs = states[sz:2*sz]
    ss = states[2*sz:3*sz]
    interval = parameters['interval']
    tidx = int(t * interval)
    if tidx >= len(fijt):
        fij = fijt[len(fijt)-1]
    else:
        fij = fijt[tidx]
    uterm = (us * population).dot(fij) / population - us * np.sum(fij,1)
    sterm = (ss * population).dot(fij) / population - ss * np.sum(fij,1)
    beta = parameters['beta']
    t_c = parameters['tc']
    t_r = parameters['tr']
    alphas = parameters['alphas']
    cross_term = beta * us * ss * intervention(parameters, t)
    #cross_term = beta * us * ss
    delta_u = cross_term - us / t_c + uterm
    delta_c = alphas * us / t_c - cs / t_r 
    delta_s = - cross_term + sterm
    domestic = cross_term
    virus_influx = (us * population).dot(fij) / population
    output = np.r_[delta_u, delta_c, delta_s, virus_influx, domestic]
    #records1.append(np.r_[delta_u, delta_c, delta_s])
    #records2.append(np.r_[us, cs, ss])
    return output
def single_np_ode(state, t, parameters, population):
    us = state[0] # un-comfirmed cases?
    cs = state[1] # comfirmed cases?
    ss = state[2] # suspectable cases?
    beta = parameters['beta']
    t_c = parameters['tc']
    t_r = parameters['tr']
    alpha = parameters['alpha']
    cross_term = beta * us * ss * intervention(parameters, t)
    
    delta_u = cross_term - us / t_c
    delta_c = alpha * us / t_c - cs / t_r 
    delta_s = - cross_term
    output = np.r_[delta_u, delta_c, delta_s]
    return output
    

class COVID19:
    def __init__(self, flowdata='data/country_flow/', casedata='data/global0415_en.csv', \
                 populationdata='data/country_population.csv', print_t0s=False, loadpath=False):
        # 首先，我们要校准各个国家的名称，做法是分别把航空流量数据、国家人口数据和病例数据中的国家字段都加载进来，统一转换为标准国家名称
        # flow data
        if not loadpath:
            self._name_combine(flowdata, casedata, populationdata)
            self._read_pop(populationdata)
            self._read_flow_dynamics(flowdata)
            self._read_case_data(casedata)
            self._find_t0s(print_t0s)
        else:
            if loadpath=='auto':
                self.loaddata('all_data_4_15.pkl')
            else:
                self.loaddata(loadpath)
        self.simulateQ = False
        self.compute_influence_matrixQ=False

    def loaddata(self, loadpath):
        f=open(loadpath, 'rb')
        output = pickle.load(f)
        f.close()
        self.all_cumconfirm_cases = output['cases'][0]
        self.all_cumexist_cases = output['cases'][1]
        self.first_cases = output['cases'][2]
        self.first_date = output['cases'][3]
        self.start_date = output['cases'][4]
        self.time_cases = output['cases'][5]
        self.nodes = output['nodes']
        self.population = output['population']
        self.fijt = output['fijt'] #t时刻流量矩阵
        
    def loadfit(self, path='logs/inidividual_parameters_4_15_log_lambda.pkl'):
        # 导入拟合参数
        f=open(path, 'rb')
        self.aaa = pickle.load(f)
        f.close()
        
    def fitted_data(self):
        return self.aaa
    
    def save_data(self, path='all_data_4_15.pkl'):
        output = {'cases':[self.all_cumconfirm_cases, self.all_cumexist_cases, \
                           self.first_cases, self.first_date, self.start_date, self.time_cases],\
                  't0s': self.t0s, 'nodes': self.nodes, \
                  'population': self.population, 'fijt': self.fijt}
        f = open(path, 'wb')
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    def compute_influence_matrix(self):
        # 计算任意国家到任意国家的病例输出/输出数据
        if not self.simulateQ:
            print('run self.simulate() before run self.compute_influence_matrix()')
            return -1
        self.influence_matrix = []
        for t in range(self.result.shape[0]):
            fluxmatrix = [row * self.population[i]  for i,row in enumerate(self.fijt[t])] #i国人口 * 流出比例 --> 绝对流出人口
            aa=np.transpose(np.repeat(np.array([self.result[t,:len(self.nodes)]]), len(self.nodes), axis=0)) # 未确诊感染比例
            i2j_virus_flux = aa*fluxmatrix # aa[i,j] * fijt[t][i,j] = aa[i] * fijt[t][i-->j]
            # i2j_virus_flux[i-->j] = i国向j国输出的病例数
            self.influence_matrix.append(i2j_virus_flux)
        self.influence_matrix = np.array(self.influence_matrix) # 转成numpy格式
        self.compute_influence_matrixQ=True #标记为已计算
    
    def plot_exports(self, names=['United States', 'United Kingdom']):
        if not self.compute_influence_matrixQ:
            print('run compute_influence_matrix() before run plot_influence_matrix()')
            return -1
        plots=[[] for i in names]

        for t in range(len(self.influence_matrix)):
            for i, name in zip(range(len(names)),names):
                plots[i].append(self._export(name, t))
        for name, ploty in zip(names,plots):
            plt.semilogy(ploty, label=name)
        plt.legend(loc='upper right', shadow=True, numpoints = 1,fontsize=10)
        plt.show()
    
    def plot_imports(self, names=['United States', 'United Kingdom']):
        if not self.compute_influence_matrixQ:
            print('run compute_influence_matrix() before run plot_influence_matrix()')
            return -1
        plots=[[] for i in names]

        for t in range(len(self.influence_matrix)):
            for i, name in zip(range(len(names)),names):
                plots[i].append(self._import(name, t))
        for name, ploty in zip(names,plots):
            plt.semilogy(ploty, label=name)
        plt.legend(loc='upper right', shadow=True, numpoints = 1,fontsize=10)
        plt.show()
    
    def _export(self, name, t):
        if not self.compute_influence_matrixQ:
            print('run compute_influence_matrix() before run this')
            return -1
        return self.influence_matrix[t][self.nodes[name],:].sum()

    def _import(self, name, t):
        if not self.compute_influence_matrixQ:
            print('run compute_influence_matrix() before run this')
            return -1
        return self.influence_matrix[t].T[self.nodes[name],:].sum()
        # TODO
        #return -1

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

        self.time_cases = np.arange(day_len) - (self.first_date - self.start_date).days
        self.all_cumconfirm_cases = np.cumsum(all_confirmed_cases, 0) - np.cumsum(all_cued_cases, 0) - np.cumsum(all_death_cases, 0)
        self.all_cumexist_cases = np.cumsum(all_confirmed_cases, 0)
        for i in range(self.all_cumconfirm_cases.shape[1]):
            yy = self.all_cumconfirm_cases[:, i]
            bools = yy>0
            plt.semilogy(self.time_cases[bools], yy[bools],'.')
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
    def _find_t0s(self, print_t0s):
        # 从现存确诊病例曲线上，推测出曲线开始下降的时间点，每个国家都不同
        self.t0s = np.ones(len(self.nodes)) * self.all_cumconfirm_cases.shape[0]
        for c,i in self.nodes.items():
            #flushPrint(i)
            curve = self.all_cumconfirm_cases[:,i]

            if max(curve)>0:
                maxv = max(curve)
                indx = np.nonzero(curve == maxv)[0]
                if print_t0s:
                    print(c,indx)
                    plt.semilogy(curve,'.')
                if indx[-1] < len(curve) - 1:
                    xs = np.ones(100) * indx[-1]
                    ys = np.logspace(0, 5, 100)
                    self.t0s[i] = indx[-1] - 5
                    if print_t0s:
                        plt.semilogy(xs, ys)
                if print_t0s:
                    plt.show()
        #输出一个数组t0s，记录了每个国家开始下降的时间点，如果没有下降，则为最后的时间点。
    def clear_simulation(self):
        del self.all_trajectories
        gc.collect()
        
    def simulate(self):
        # 模拟所有国家的感染
        t_c = 8.3
        t_r = 9.2
        r0 = 2.3
        initials = np.zeros(len(self.nodes))
        alphas = np.zeros(len(self.nodes))
        betas = np.zeros(len(self.nodes))
        lambds = np.zeros(len(self.nodes))
        t0s = np.zeros(len(self.nodes))
        for i,(c,v) in enumerate(self.aaa.items()):
            initials[self.nodes[c]] = v['initial_unconfirmed'][0]
            alphas[self.nodes[c]]=v['alphas'][0]
            betas[self.nodes[c]]=v['betas'][0]
            t0s[self.nodes[c]]=v['t0s'][0]
            lambds[self.nodes[c]]=v['lambds'][0]

        self.all_trajectories = np.zeros([self.all_cumconfirm_cases.shape[0], self.all_cumconfirm_cases.shape[1], 3])
        epidemic_start_time = np.zeros(len(self.nodes))
        for c,i in self.nodes.items():
            #print(c)
            cumconfirm = np.nonzero(self.all_cumconfirm_cases[:,i])[0]
            if len(cumconfirm)> 0 :
                first_t = cumconfirm[0]
                epidemic_start_time[i] = first_t
                timeline = len(self.time_cases) - first_t
                timespan = np.linspace(0, timeline, timeline)
                cases_data = self.all_cumconfirm_cases[first_t:, i]
                cs0 = cases_data[0] / self.population[i]
                us0 = initials[i] / self.population[i]
                ss0 = 1 - cs0 - us0
                initial_state = np.r_[us0, cs0, ss0]
                t0ss = t0s[i] - first_t
                constants = {'beta': r0 / t_c, 'tc': t_c, 'tr': t_r, 'interval': 1,'tstar':0,'epsilon':0.001,'lambds2':0}
                constants['beta'] = betas[i]
                constants['alpha'] = alphas[i]
                constants['t0s'] = t0s[i]
                constants['lambds'] = lambds[i]

                result = odeint(single_np_ode, initial_state, timespan, args = (constants, self.population))
                partt = alphas[i] * result[:, 0] / (constants['tc'] * constants['interval'])
                partt = np.cumsum(partt)
                self.all_trajectories[first_t:len(self.time_cases), i, 0:2] = result[:, 0:2]
                self.all_trajectories[first_t:len(self.time_cases), i, 2] = partt
                #plt.semilogy(first_t + timespan, result[:, 1] * self.population[i])
                #plt.semilogy(np.linspace(1,len(self.time_cases), len(self.time_cases)), self.all_cumexist_cases[:, i],'.')
                #plt.semilogy(first_t + timespan, partt * self.population[i])
                #plt.semilogy(np.linspace(1,len(self.time_cases), len(self.time_cases)), self.all_cumconfirm_cases[:, i],'.')
                #plt.show()

        #输出一个数组all_trajectories，格式time_length*国家数*3，记录了任意时刻任意国家的未确诊、确诊和累积现存病例数
        # 以all_trajectories最后一个时刻所记录的数据作为初始条件，开始模拟每个国家的增长情况
        t_c = 8.3
        t_r = 9.2
        r0 = 2.3

        timespan = np.linspace(len(self.time_cases), len(self.fijt), len(self.fijt) - len(self.time_cases))
        constants = {'beta': r0 / t_c, 'tc': t_c, 'tr': t_r, 'interval': 1,'tstar':0,'epsilon':0.001,'lambds2':0}


        us00 = self.all_trajectories[-1, :,0]
        cs00 = self.all_trajectories[-1, :,1]
        ss00 = 1 - us00 - cs00
        virus_influx0 = np.zeros(len(self.nodes))
        domestic_flux0 = np.zeros(len(self.nodes))
        initial_states = np.r_[us00,cs00,ss00,virus_influx0,domestic_flux0]
        params = constants
        params['beta'] = betas
        params['alphas'] = alphas
        params['t0s'] = t0s + epidemic_start_time
        params['lambds'] = lambds
        self.result = odeint(np_ode, initial_states, timespan, args = (params, self.fijt, self.population))
        self.simulateQ = True


    def plot_simulation(self,\
                        countries=['China','United States','United Kingdom','France','Italy','Spain','Iran','Japan','Mexico','South Africa']):
        # 挑选了若干个代表性国家，绘制它们的疫情增长曲线
        plt.figure(figsize = (15,10))
        colors = plt.cm.jet(np.linspace(0,1,len(countries)))
        plot_time = np.linspace(0, len(self.fijt), len(self.fijt)-1)

        for i,country in enumerate(countries):
            row = self.nodes.get(country, -1)
            pop = self.population[self.nodes[country]]
            #try:
            show = np.r_[self.all_trajectories[:, row, 1] ,self.result[1:, len(self.nodes)+row]]
            show = show * pop
            #show = result[:,  row] * pop
            plt.semilogy(plot_time-(self.first_date - self.start_date).days, show, '-', color = colors[i], label=country)
            plt.semilogy(np.arange(self.all_cumconfirm_cases.shape[0])-(self.first_date - self.start_date).days
                         , self.all_cumconfirm_cases[:, row], 'o', color = colors[i])


        plt.legend(loc='upper left', shadow=True, numpoints = 1,fontsize=10)
        plt.ylim([1, 10**10])
        plt.show()