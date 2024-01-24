# require python >= 3.7

import numpy as np
import random
import matplotlib.pyplot as plt
import copy

import sys




class client:
    def __init__(self, id, Ynt, dist,charge,bandwidth):
        # 客户端ID
        self.id = id
        # 客户端计算资源
        self.Ynt = Ynt
        # 客户端离ES距离
        self.dist = dist
        # 资源定价
        self.charge = charge
        # 客户端带宽
        self.bandwidth = bandwidth
        # 初始化客户端信道状态
        self.Cdt = 0
        # 初始化Context
        self.context = 0

        self.refresh_Cdt()
        self.refresh_context()

    def refresh_context(self):
        self.context = np.array([self.Cdt])

    def refresh_Cdt(self):
        gDt = (37.6) * np.log(self.dist) + 128.1
        # 噪声功率
        No = 12000
        # No = random.randint(10000, 14000)
        self.Cdt = np.log2(1 + gDt * 23 / No)
        


class CCMAB:

    def __init__(self, num_rounds, id, num_context_features):
        self.num_context_features = num_context_features
        self.num_arms = 1 # default number of arms
        
        # ES的编号
        self.id = id
        self.hT = 5
        self.num_round = num_rounds
        # self.hT = np.ceil(self.num_rounds ** (1 / (3 + num_context_features)))
        
        self.cube_length = 1 / self.hT
        self.budget = 7
        self.alpha = 1
        
        # 概率字典 {(n客户端id,l上下文cube):0.3,(1,(0.2)):0.5, (2,(0.4)):0.7}   每轮结束时根据"计数器"和"成功参加的客户端"更新(context):p
        self.p_dict = {}

        # 范围内的客户端对象列表
        self.available_clients = []
        self.available_clients_id_list = []

    def knapsack_01(self, weights, values, capacity, scale_factor=100):
        n = len(values)

        # 放大scale_factor倍
        scaled_values = [int(val * scale_factor) for val in values]
        scaled_weights = [int(weight * scale_factor) for weight in weights]
        scaled_capacity = int(capacity * scale_factor)

        
        # 创建一个二维数组来保存子问题的解
        dp = [[0 for _ in range(scaled_capacity + 1)] for _ in range(n + 1)]

        # 填充dp数组
        for i in range(1, n + 1):
            for w in range(scaled_capacity + 1):
                # 如果当前物品的重量超过当前背包容量，无法放入
                if scaled_weights[i - 1] > w:
                    dp[i][w] = dp[i - 1][w]
                else:
                    # 考虑将当前物品放入或不放入背包，选择价值最大的情况
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - scaled_weights[i - 1]] + scaled_values[i - 1])

        # 通过dp数组回溯找到选择的物品索引
        selected_items = []
        i, w = n, scaled_capacity
        while i > 0 and w > 0:
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(i - 1)
                w -= scaled_weights[i - 1]
            i -= 1

        total_value  = dp[n][scaled_capacity]

        # 映射回原始浮点数值
        selected_items_original = [val / scale_factor for val in selected_items]
        total_value_original = total_value / scale_factor

        # 返回结果，包括选择的物品索引和总价值
        return  total_value_original, selected_items[::-1]


    # weights 计算机资源代价数组
    # values 预估概率数组
    # capacity 背包容量
    def knapsack_greedy(self, weights, values, capacity):
        n = len(weights)

        # 计算单位价值和剩余容量的比例
        ratios = [(values[i] / weights[i], i) for i in range(n)]

        # 按照单位价值和剩余容量比例排序
        sorted_items = sorted(ratios, key=lambda x: x[0], reverse=True)

        # 初始化列表用于存储被选中的物品ID
        selected_item_ids = []

        # 初始化变量用于跟踪背包容量
        remaining_capacity = capacity

        # 贪心选择，优先选择单位价值和剩余容量比例高的物品
        result = 0
        for _, i in sorted_items:
            if weights[i] <= remaining_capacity:
                # 将整个物品放入背包
                selected_item_ids.append(i)
                remaining_capacity -= weights[i]
                result += values[i]

        return result, selected_item_ids


     # 将上下文映射到超立方体
    
    # weights 计算机资源数组
    # values 预估概率数组
    # capacity 背包容量
    # return result(p累加最大), item_list(索引列表) 
    def knapsack_bruteforce(self, weights, values, capacity, current_item):
        if capacity <= 0 or current_item < 0:
            return 0, []

        # 不选择当前物品
        result_without_current, items_without_current = self.knapsack_bruteforce(weights, values, capacity, current_item - 1)

        # 选择当前物品
        result_with_current, items_with_current = 0, []
        if weights[current_item] <= capacity:
            result_with_current, items_with_current = self.knapsack_bruteforce(
                weights, values, capacity - weights[current_item], current_item - 1
            )
            if current_item > 0 :
                result_with_current += values[current_item]
                items_with_current.append(current_item)

        # 返回两种情况中的最大值和相应的物品列表
        if result_with_current > result_without_current:
            return result_with_current, items_with_current
        else:
            return result_without_current, items_without_current


    
    def get_cube_context(self, context):
        return tuple((context * self.cube_length).astype(int))

    # 获得识别元组
    def get_cube_of_client_and_context(self, client_id, cube_context):
        return tuple((client_id, self.id, cube_context))

    def get_cube_of_id_and_context(self, client_id, cube_context):
        return tuple((client_id, cube_context))

    # 获取范围内的客户端对象列表
    def get_available_clients(self,available_clients):
        self.available_clients = available_clients

    # 根据范围内的客户端，返回符合约束的客户端集, counter 关于client-es-contextCube对的全局计数器
    # counter = {(n,m,(context cube)):int,(1,2,(0.2)):0.2}
    def get_selected_clients(self, counter_dict, t, model):
        
        arrived_cube_clients_dict = {}

        arrived_cube_set = set()

        # 统计所有可用客户端的上下文(若有未识别的,加入识别的列中)
        for available_client in self.available_clients:
            # 获得识别元组 identity_cube = (n,m,(context cube))  
            identity_cube = self.get_cube_of_client_and_context(available_client.id, self.get_cube_context(available_client.context))
            if identity_cube not in arrived_cube_clients_dict:
                arrived_cube_clients_dict[identity_cube] = list()
            
            # 以下一行暂时没什么用
            arrived_cube_clients_dict[identity_cube].append(available_client)
            arrived_cube_set.add(identity_cube)

        # Identify 识别阶段:识别所有的被定为"未探索的"客户端    
        underexplored_cube_set = set()
        # 遍历每个被记录的识别元组identify_cube, cube = (n,m,(context))
        for cube in arrived_cube_set:
            # 如果计数器上(n,m,context)对应计数小于 K(T)
            if counter_dict.get(cube,0) <= t ** (( 2*self.alpha )/( 3 * self.alpha +  2 )) * np.log(t):
                # 此处不能使用update
                underexplored_cube_set.add(cube)


        # 开始进行选择
        selected_clients_id_list = []
        # 获得所有可用客户端的id列表，收取费用列表，索引对齐
        self.available_clients_id_list = [obj.id for obj in self.available_clients]
        # available_clients_cost_list = [(0.18 * obj.charge / 8) for obj in self.available_clients]
        available_clients_cost_list = [(obj.Ynt * obj.charge / 8 ) for obj in self.available_clients]

        # 根据available_clients的客户端信息，按顺序整理出对应的概率字典
        clients_p_dict = {} # 索引为客户端编号
        for obj in self.available_clients:
            key = (obj.id, self.get_cube_context(obj.context))
            if key in self.p_dict:
                clients_p_dict.update({obj.id:self.p_dict[key]})
            else:
                clients_p_dict.update({obj.id:0})
                pass
                
        # 模式判断 若为随机则在可选范围内进行随机选择
        if model == 'Random':
            # 法1：随机选出符合采样率数量
            # selected_clients_id_list = np.random.choice(self.available_clients_id_list, size=int(len(self.available_clients_id_list)*1/10), replace=False)
            # selected_clients_cost_list = []
            # cost_sum = 0
            # # 整理出所选择客户端的花费列表，索引对齐
            # for i in range(len(self.available_clients_id_list)):
            #     for cid in selected_clients_id_list:
            #         if cid == self.available_clients_id_list[i]:
            #             selected_clients_cost_list.append(available_clients_cost_list[i])
            #             cost_sum += available_clients_cost_list[i]
            # # 检查所选择的结果是否符合budget
            # print('cost_sum:',cost_sum)
            # if cost_sum > self.budget:
            #     # 不符合budget则一随机个个删除元素
            #     while(True):
            #         del_id = random.choice(selected_clients_id_list)
            #         del_index = selected_clients_id_list.index(del_id)
            #         # two way to delete an element 
            #         cost_sum -= selected_clients_cost_list[del_index]
            #         selected_clients_id_list.remove(del_id)
            #         del selected_clients_cost_list[del_index]
            #         if cost_sum <= self.budget :
            #             break
                    
            # 法2：一个个选出来
            temp_list = copy.deepcopy(self.available_clients_id_list)
            selected_clients_cost_list = []
            cost_sum = 0
            for i in range(int(len(self.available_clients_id_list)*0.5)):
                temp_id = np.random.choice(temp_list, size=1,replace=False)
                temp_cost = available_clients_cost_list[i]
                if cost_sum + temp_cost <= self.budget:
                    selected_clients_id_list.append(temp_id[0])
                    cost_sum += available_clients_cost_list[i]  
                    temp_list.remove(temp_id)
                else:
                    break
            selected_clients_id_list = np.array(selected_clients_id_list)
            return len(selected_clients_id_list), selected_clients_id_list.tolist()
        else:
            pass

        # clients_p_list = [self.p_dict[(obj.id, self.get_cube_context(obj.context))] for obj in self.available_clients]
        
        # 如果当前所有的客户端(n,m,context)都没探索过,没探索过的标准：
        # n换了位置,导致m或context变了,并且原来没有记录
        if len(underexplored_cube_set) == len(arrived_cube_set):
            # TODO (14)
            #  全部都没搜索过
            #  0.18 dataSize 
            clients_dict = {obj.id: (obj.Ynt * obj.charge / 8)  for obj in self.available_clients}
            clients_dict_Incr = dict(sorted(clients_dict.items(),key=lambda item:item[1]))

            # 按顺序遍历字典
            sum = 0
            for key,value in clients_dict_Incr.items():
                sum += value
                if sum + value <= self.budget:
                    sum += value
                    selected_clients_id_list.append(key)
                else:
                    break

            # print("1 Explore Result: ", 0, "selected_list:", selected_clients_id_list)

        # 当前所有的客户端都已经被探索过
        elif len(underexplored_cube_set) == 0:
            # TODO: (18)

            clients_p_list = []
            for client in self.available_clients :
                #  获得每个客户端对应的概率的数组
                p = self.p_dict.get(self.get_cube_of_id_and_context(client.id,self.get_cube_context(client.context)))
                clients_p_list.append(p)

            # 暴力搜寻结果
            # result, selected_clients_index_list = self.knapsack_bruteforce(available_clients_cost_list, clients_p_list, self.budget, len(available_clients_cost_list) - 1)
            # 贪婪
            # result, selected_clients_index_list = self.knapsack_greedy(available_clients_cost_list, clients_p_list,self.budget)
            # 动态规划
            result, selected_clients_index_list = self.knapsack_01(available_clients_cost_list, clients_p_list, self.budget)
            

            # 将所选择的客户端的索引，转化成客户端id数组
            selected_clients_id_list = []
            for index in selected_clients_index_list:
                selected_clients_id_list.append(self.available_clients_id_list[index])

            # print("2 Explored Brute-force result: ", result, "selected_list:", selected_clients_id_list)

        # 部分被探索过，优先选择未探索的，剩下的预算用来选择已探索的
        else:
            # TODO: (17)

            # clients_dict = {client.id: client.charge} 根据id：价格做成字典
            clients_dict = {obj.id: (obj.Ynt * obj.charge / 8 ) for obj in self.available_clients}
            # 升序排列
            clients_dict_Incr = dict(sorted(clients_dict.items(),key=lambda item:item[1]))

            underexplored_clients_id_list = []
            selected_clients_id_list = []

            # 按花费顺序从小到大收集未探索的客户端，同时计算花费预算防止超预算 
            explore_cost = 0
            for k,v in clients_dict_Incr.items():
                for cube in underexplored_cube_set:
                    # cube[0]为 client的 id
                    if cube[0] == k :
                        cost = clients_dict.get(cube[0])
                        if explore_cost + cost <= self.budget:
                            explore_cost += cost
                            selected_clients_id_list.append(cube[0])
                        else :
                            break
                if explore_cost >= self.budget :
                    break
            
            # print("3-1 Part Explored Brute-force result: ",len(selected_clients_id_list)," selected_list:",selected_clients_id_list)

            # 若还剩下预算，则对已探索部分进行挑选 
            if explore_cost < self.budget :     
            
                # 有一部分的id ,求剩下未选客户端的id,charge, 索引对齐
                remain_budget = self.budget - explore_cost
                remain_clients_id_list = []
                remain_clients_cost_list = []
                remain_clients_p_list = []
                # 将已探索的客户端收集
                for i in range(len(self.available_clients_id_list)):
                    if self.available_clients_id_list[i] not in selected_clients_id_list:
                        remain_clients_id_list.append(self.available_clients_id_list[i]) 
                        # 计算cost_list
                        remain_clients_cost_list.append(available_clients_cost_list[i])
                        # 按顺序，读入对应的概率，各list索引对齐
                        if self.available_clients_id_list[i] in clients_p_dict:
                            remain_clients_p_list.append(clients_p_dict[self.available_clients_id_list[i]])
                        
                # 暴力求解，得到概率累加值，所选的客户端的index
                # result, selected_clients_index_list = self.knapsack_bruteforce(remain_clients_cost_list, remain_clients_p_list, remain_budget, len(remain_clients_cost_list) - 1)
                # 贪婪求解
                # result, selected_clients_index_list = self.knapsack_greedy(remain_clients_cost_list, remain_clients_p_list, remain_budget)
                # 动态规划
                result, selected_clients_index_list = self.knapsack_01(remain_clients_cost_list, remain_clients_p_list, remain_budget)
            

                # 将所选择的客户端的索引，转化成客户端id数组
                for index in selected_clients_index_list:
                    selected_clients_id_list.append(remain_clients_id_list[index])

            # print("3-2 Part Explored Brute-force result: ",result," selected_list:",selected_clients_id_list)

        return len(selected_clients_id_list), selected_clients_id_list
    

    def update(self,successful_participants_id,counter_dict):
        for id in successful_participants_id:
            for client in self.available_clients:
                if client.id == id :
                    # cube = (n,m,(context))
                    cube = self.get_cube_of_client_and_context(client.id, self.get_cube_context(client.context))
                    new_counter = counter_dict[cube] = counter_dict.get(cube,0) + 1

                    # 进行概率 p更新
                    # p_cube = (n,(context))
                    # p_cube = self.get_cube_of_id_and_context(id,self.get_cube_context(client.context))
                    p_cube = (cube[0], cube[2])
                    self.p_dict[p_cube] = (self.p_dict.get(p_cube, 0) * (
                        new_counter - 1) + 1) / new_counter
                        
        return 0 


def random_selected(client_list):
    pass




if __name__ == "__main__":

    # 定义上下文维度
    context_dim = 1

    # 定义客户端数量
    num_clients = 26

    num_rounds = 200

    # 定义客户端离ES的距离dist的分布, 0.04防止计算g得负数
    dist_distribution = np.random.uniform(low=0.04, high=2, size=num_clients)

    # 定义客户端的资源定价charge的分布 (论文对应Pricing funciton)
    charge_distribution = np.random.uniform(low=0.5, high=2, size=num_clients)

     # 定义客户端的计算资源y的分布
    y_distribution = np.random.uniform(low=2, high=4, size=num_clients)

    # 定义客户端信道状态Cdt的分布
    # cdt_distribution = np.random.uniform(0,1,size=num_clients)

    # 定义客户端带宽的分布
    bandwidth_distribution = np.random.uniform(low=0.3, high=1, size=num_clients)

    # 生成客户端对象列表
    clients = []
    for i in range(num_clients):
        clients.append(
            client(
                i,
                # cdt_distribution[i],
                y_distribution[i],
                dist_distribution[i],
                charge_distribution[i],
                bandwidth_distribution[i]
            )
        )

    # 定义ES的编号
    es_id = 1

    # 定义CC-MAB算法实例
    cc_mab = CCMAB(num_rounds=num_rounds, id=es_id, num_context_features=context_dim)

    # 获取范围内的客户端对象列表
    cc_mab.get_available_clients(clients)

    # 定义计数器
    counter = {}

    # 定义画图数组
    len_counter = []
    all_counter = np.array([0] * num_rounds)

    # 开始迭代
    for r in range(3):
        for round in range(num_rounds+1):

            cc_mab.get_available_clients(clients)

            # 获取选择的客户端
            # _, selected_clients_id_list = cc_mab.get_selected_clients(counter, round+1,'normal')
            _, selected_clients_id_list = cc_mab.get_selected_clients(counter, round+1, sys.argv[1])

            print("selected client :", selected_clients_id_list)
            # 对获取的客户端模拟：是否能在deadline前获得
            Cdt_min = 37.6 * np.log(0.001) + 128.1
            Cdt_max = 37.6 * np.log(2) + 128.1
            for id in selected_clients_id_list:
                for client in clients:
                    if client.id == id:
                        # TODO: Cdt应该根据当前round所有值作归一化
                        # client.Cdt =(37.6 * np.log(client.dist) + 128.1 - Cdt_min) / (Cdt_max - Cdt_min)
                    
                        tDT = 0.18 / (client.bandwidth * client.Cdt )
                        tLC = 2.41 / client.Ynt
                        tUT = tDT
                        # print("===time:",tDT + tLC + tUT," tDT:",tDT," tLC:",tLC," Ynt:",client.Ynt," cDT",client.Cdt," gDT",gDt," Dist",client.dist)
                        if tDT + tLC + tUT > 4 :
                            # print("===time:",tDT + tLC + tUT)
                            selected_clients_id_list.remove(id)

            print("reached client :", selected_clients_id_list)


            # 更新算法状态
            cc_mab.update(selected_clients_id_list, counter)

            # 更新客户端状态
            # 动态模拟模块(更新下一轮状态)
            for client in clients:
                p = random.random()
                if p < 0.05:
                    # 选择某部分刷新距离
                    client.dist = np.random.uniform(low=0.04, high=2)
                    client.Ynt = np.random.uniform(low=2, high=4)
                    client.bandwidth = np.random.uniform(low=0.3, high=1)
                    client.refresh_Cdt()
                    client.refresh_context()
                    # 测试
                    # client.charge = np.random.uniform(0.5, 2)
            
            # 用于画图
            len_counter.append(len(selected_clients_id_list))

        all_counter = [a + b for a, b in zip(all_counter, len_counter)]

    all_counter = [x / 1 for x in all_counter]
    time_list = list(range(1,round+1))
    plt.plot(all_counter, label=sys.argv[1])
    plt.xlabel("round")
    plt.ylabel("number of clients")
    plt.legend()
    plt.show()

    np.savetxt(sys.argv[1]+'-result.csv', all_counter, delimiter=',')
    
    # 算方差
    a = all_counter = np.array([all_counter])
    var_a = np.var(a)
    print(var_a)
    # 输出结果
    # print("selected_clients_id_list:", selected_clients_id_list)