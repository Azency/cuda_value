import pandas as pd
import numpy as np
import math



class D_Prob():
    def __init__(self) -> None:
        self.ax = None
        self.bx = None
        self.kt = None

    def read_abk(self, path_list:list):
        '''
        path_list : ax, bx, kt存储的路径
        
        '''
        if len(path_list) != 3:
            raise Exception("至少要三个路径！")
        self.ax = pd.read_csv(path_list[0])['x'].tolist()
        self.bx = pd.read_csv(path_list[1])['x'].tolist()
        self.kt = pd.read_csv(path_list[2])["Point.Forecast"].tolist()

#np.exp(-self._gen_mxt(x0+i, T0+i, T0)），x0+i岁的人在当年的存活率 ，i=1,2,3,...,t-1,T
    def _gen_mxt(self, x0, T, T0) -> float:
        '''
        需要mxt的调用这个值
        x0 : 年龄
        T : 公元年份
        T0:投保的公元年份
        '''
        a = np.exp(self.ax[x0] + self.bx[x0]*self.kt[T-T0-1])
        #真实的年龄，没有加1，因为ax本身序列索引也是从0开始（0岁，1岁，...）
        # print(f"self.ax[x0]:{self.ax[x0]}且x0:{x0}")
        # print(f"self.bx[x0]:{self.bx[x0]}且x0:{x0}")
        # print(f"self.kt[T-T0-1]:{self.kt[T-T0-1]}且序列是:{T-T0-1}")
        # print(a)

        return a

###########################################################################################################################
#t等于几，就连乘几个存活且不退保的概率，就是几天
    def accu_live_nosur(self, x0, T0, k, t, p, sur_rate):#计算从第0天到第t天(第i年)一直活着且没有退保的概率，t>=0,t = 0时，P = 1
        P = 1
        for j in range(1,t+1): #0时刻已经放在P = 1中了；range(1,t+1) = 1，2，...,t
            i = math.ceil(k+j/p)
            P *= (np.exp(-self._gen_mxt(x0+i, T0+i, T0)))**(1/p)(1 - sur_rate[j])

        return P
#####################################################################################################################

    def first_single_live(self, x0, T0, t, p):#计算在第i年从第t天活到该年最后一天i*p天的概率，t>=0,t = 0时，P = 1
        i = math.ceil(t/p)
        P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(i*p - t)/p)
        # print(f"整数为{i}年, m_x,t:{self._gen_mxt(x0+i, T0+i, T0)}，样本点{t}对应的{t-i*p}天的零散存活概率q:{P}")

        return P
    
    def last_single_live(self, x0, T0, t, p):#计算在第i年的第一天:(i-1)*p+1天，活到第t-1天的概率
        i = math.ceil(t/p)
        if i == 0:
            P = 1
        else:
            P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(t-1-(i-1)*p)/p)

        return P

    def last_single_death(self, x0, T0, k, t, p):#计算在i年中，任意一颗粒度的死亡概率
        i= math.ceil(k+t/p)
        if i == 0:
            P = 0
        else:
            P = 1-np.exp(-self._gen_mxt(x0+i, T0+i, T0)/p) #最后一天的死亡率

        return P
 

    def gen_P(self, x0, T0, t):#计算从0时刻起到第t整数年死亡的条件概率
        '''
        生成对应的概率
        x0 : 投保的年龄 
        T0 : 投保的公元年份,eg2020
        t : 死亡的时间
        '''
        P = 1-np.exp(-self._gen_mxt(x0+t, T0+t, T0)) #最后一年的死亡率
        for i in range(t-1, 0, -1):#倒退计算，最后一年之前的累计存活概率  ##############是否需要计算到-1年，否，因为kt序列从0开始索引
            P *= np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))

        # P = 2*self._gen_mxt(x0+t, T0+t, T0)/(2 + self._gen_mxt(x0+t, T0+t, T0)) #最后一年的死亡概率
        # for i in range(t-1, 0, -1):#倒退计算，最后一年之前的累计存活概率
        #     P *= 1 - 2*self._gen_mxt(x0+i, T0+i, T0)/(2 + self._gen_mxt(x0+i, T0+i, T0))

        return P

#######################
# #代码中的计算逻辑和文章中写的一致，interval_death_P和unit_P不同
#interval_death_P 从第t（t>0）个计数单元到第s（s>1)个计数单元条件死亡概率
#unit_P 从第t=0个计数单元到第s（s>=0)个计数单元条件死亡概率      
    def unit_P(self, x0, T0, t, p):#计算从购买合同 0时刻起，在第t个计数单元刚好死亡的条件概率，t>=1
        P = self.last_single_death(x0, T0, t, p)
        # print("last_single_death",P)
        P *= self.last_single_live(x0, T0, t, p)#t天前这一年内活着
        # print("last_single_live",self.last_single_live(x0, T0, t, p))
        P *= self.accu_live(x0, T0, math.ceil(t/p)-1, 0)
        # print("accu_live",self.accu_live(x0, T0, math.ceil(t/p)-1, 0))

        return P
    

    def interval_death_P(self, x0, T0, t, s, p):#计算从t个计数单元起到第s个计数单元时死亡的条件概率,s>t>0严格
        i = math.ceil(s/p)
        j = math.ceil(t/p)

        if j == i:
            P = self.last_single_death(x0, T0, s, p)*np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(s-1-t)/p)
        else:
            P = self.last_single_death(x0, T0, s, p)
            # print("single_death",P)
            P *=self.last_single_live(x0, T0, s, p)
            # print("single_live", self.last_single_live(x0, T0, s, p), P)
            P *= self.accu_live(x0, T0, i-1, j)
            # print("accu_live",self.accu_live(x0, T0, i-1, j), P)
            P *= self.first_single_live(x0, T0, t, p)
            # print("interval_single_live",self.first_single_live(x0, T0, t, p), P)

        return P
    
#accu_live可以从0开始计数
#interval_terminal_live 必须从1开始计数
    def accu_live(self, x0, T0, t, s):#计算从s+1年起到第t年的累计存活概率,其中t最多是T,s>=0.当s= 0，序列取值是：t,t-1,t-2,...,1
        if t == 0:
            P_a = 1
        else:
            P_a = 1#投保前存活概率
            for i in range(t, s, -1):#倒退计算，t,t-1,t-2,...,1
                P_a *= np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))  #x0+i岁的人在当年的存活率 ，i=t,t-1,...1 
                # print(f"第{i}年存活率：{np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))}")    

        return P_a
    
    def interval_terminal_live(self, x0, T0, t, T, p):#计算从t时刻起到最后时刻的累计存活概率,t>=1
        j = math.ceil(t/p)

        P = self.accu_live(x0, T0, T, j)##accu_live函数，T，T-1，...,j+1年存活率累积
        # print(f"j+1到T的整数年存活率{P}")
        P *= self.first_single_live(x0, T0, t, p)
        # print(f"第一年零散的存活率：{self.first_single_live(x0, T0, t, p)}")

        return P
    

###############################################
#unit_live从0开始计数
#interval_live_P从1开始计数
    
    def unit_live(self, x0, T0, t, p):#计算从购买合同0时刻起存活到第t（t>=1）个计数单元的概率，
        i = math.ceil(t/p)
        if i == 0:
            P = 1
        else:
            P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(t-(i-1)*p)/p)
        P *= self.accu_live(x0, T0, math.ceil(t/p)-1, 0)
        return P
    
    def interval_live_P(self, x0, T0, t, s, p):#计算从t时刻起到第s个计数单元时累计存活概率
        i = math.ceil(s/p)
        j = math.ceil(t/p)

        if j == i:
            P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(s-t)/p)
        else:
            P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(s-(i-1)*p)/p)
            P *= self.accu_live(x0, T0, i-1, j)
            P *= self.first_single_live(x0, T0, t, p)

        return P