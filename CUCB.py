import numpy as np
from itertools import combinations

import random
import time 
import copy
import matplotlib.pyplot as plt
import collections



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







class CombinatorialUCB:
    def __init__(self, num_arms, num_actions):
        self.num_arms = num_arms
        self.num_actions = num_actions
        self.counts = np.zeros((num_arms, num_actions))  # Number of times each arm-action pair is chosen
        self.values = np.zeros((num_arms, num_actions))  # Estimated values for each arm-action pair
        self.total_counts = 0

    def select_arm(self):
        """Select an arm using the UCB criterion."""
        exploration_bonus = np.sqrt(2 * np.log(self.total_counts + 1) / (self.counts + 1e-6))
        ucb_values = self.values + exploration_bonus
        arm = np.argmax(np.sum(ucb_values, axis=1))
        return arm
    
    

    def update(self, arm, action, reward):
        """Update the estimates based on the observed reward."""
        # 这里注意action减一，因为索引问题
        self.counts[arm, action-1] += 1
        self.total_counts += 1
        current_value = self.values[arm, action-1]
        new_value = ((self.counts[arm, action-1] - 1) * current_value + reward) / self.counts[arm, action-1]
        self.values[arm, action-1] = new_value
        
    # 回溯两套
    def backtrack_budget_combinations(self, A, B, target_budget, current_index, current_budget, current_combination, result):
        if current_budget == target_budget:
            result.append(set(current_combination))
            return
        elif current_budget > target_budget:
            current_combination.pop()
            result.append(set(current_combination))
            return 

        for i in range(current_index, len(A)):
            if current_budget + B[i] <= target_budget:
                current_combination.append(i)
                self.backtrack_budget_combinations(A, B, target_budget, i + 1, current_budget + B[i], current_combination, result)
                current_combination.pop()

    def find_all_budget_combinations(self, A, B, target_budget):
        result = []
        self.backtrack_budget_combinations(A, B, target_budget, 0, 0, [], result)
        return list(result)
        
        
        
n = 0       
if __name__ == '__main__':
    # 定义上下文维度
    context_dim = 1

    # 定义客户端数量
    num_clients = 26

    # 摇臂的回合数
    num_rounds = 200
    # 手臂的动作 拓展用
    num_actions = 1

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
    
    
    # 假设你有一个30个元素的数组
    # original_array = tuple(range(num_clients))
    original_array = (1, 2, 9, 10, 11, 16, 17, 28, 33, 40, 43, 44, 45, 51, 55,56,61,67,68,70,71,73,75,76,78,79)
    Budget_list = tuple(round(random.uniform(0.1, 0.3), 2) for _ in range(26)) 
    # Budget_list = tuple([])
    Budget = 3.5
    
    # CUCB初始化
    num_arms = num_combinations = len(original_array)
    ucb_algorithm = CombinatorialUCB(num_arms, num_actions)
    
    # 初始化调用
    start_time = time.time()
    
    # !回溯 获得所有可能的组合
    combinations = ucb_algorithm.find_all_budget_combinations(original_array, Budget_list, Budget)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    num_arms = num_combinations = len(combinations)
    
    # 输出结果
    # print("here",np.array(initial_combinations))
    # print("Budget_list",Budget_list)
    # for combo in initial_combinations:
    #     print("Selected IDs:", [original_array[i] for i in combo])
    #     print("Total Budget:", sum(Budget_list[j] for j in combo))
    #     print("-" * 30)
        
    print(f"程序运行了 {elapsed_time:.2f} 秒")
    print("长度",len(combinations))
    
    # 对组合进行去重
    set_of_sets = set(map(frozenset, combinations))
    if len(original_array) != len(set_of_sets):
        print("数组中存在重复的集合")
        combinations = list(set_of_sets)          
        print("去重后长度",len(combinations))
    else :
        print("数组中没有重复的集合")
        
    # 以下是CUCB测试
    # initial_combinations 
    num_arms_index = list(range(1, len(combinations)))
    
    # 定义画图数组
    len_counter = []
    all_counter = np.array([0] * num_rounds)
    
    start_time = time.time()
        
    for _ in range(num_rounds):
        num_counter = 0
        for r in range(3):
            # 选择一个臂
            selected_arm = ucb_algorithm.select_arm()
            print("Select:",selected_arm)

            # 在选择的臂上选择一个动作（这里简化为直接使用臂的索引作为动作）
            selected_action = 1

            # 模拟观察到的奖励（可以根据具体情境更改此部分）
            reward = np.random.normal(loc=0.5, scale=1.0)
            
            # 抽出所选择的客户端组合(深拷贝)
            selected_clients_id_list = set(copy.deepcopy(combinations[selected_arm]))
            reached_clients_list = set(copy.deepcopy(combinations[selected_arm]))
            # 计算哪些客户端能够到达
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
                                reached_clients_list.remove(id)
            
            reward = len(reached_clients_list) / len(combinations[selected_arm])
            print("reached:", reached_clients_list)

            # 更新算法的估计值
            ucb_algorithm.update(selected_arm, selected_action, reward)
            
            for client in clients:
                    p = random.random()
                    if p < 0.1:
                        # 选择某部分刷新距离
                        client.dist = np.random.uniform(low=0.04, high=2)
                        # CUCB中不改变计算资源和带宽
                        # client.Ynt = np.random.uniform(low=2, high=4)
                        # client.bandwidth = np.random.uniform(low=0.3, high=1)
                        # client.refresh_Cdt()
                        # client.refresh_context()
            num_counter += len(selected_clients_id_list)
        # 用于画图
        len_counter.append(num_counter)
    
    end_time = time.time()
    print(f"程序运行了 {end_time - start_time:.2f} 秒")
    
    all_counter = [a + b for a, b in zip(all_counter, len_counter)]
    plt.plot(len_counter, label="CUCB")
    plt.xlabel("round")
    plt.ylabel("number of clients")
    plt.legend()
    plt.show()
    
    np.savetxt('CUCB-result.csv', len_counter, delimiter=',')
    
     # 算方差
    a = all_counter = np.array([all_counter])
    var_a = np.var(a)
    print(var_a)