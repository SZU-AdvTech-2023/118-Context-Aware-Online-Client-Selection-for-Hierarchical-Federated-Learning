# Flow of the algorithm
# Client update(t_1) -> Edge Aggregate(t_2) -> Cloud Aggregate(t_3)

from options import args_parser
from tensorboardX import SummaryWriter
import torch
from client import Client
from edge import Edge
from cloud import Cloud
from datasets.get_data import get_dataloaders, show_distribution
import copy
import numpy as np
from tqdm import tqdm
from models.mnist_cnn import mnist_lenet
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18
from models.mnist_logistic import LogisticRegression
import os

from My_CCMAB import CCMAB
from CUCB import CombinatorialUCB
import random
import matplotlib.pyplot as plt



def get_client_class(args, clients):
    client_class = []
    client_class_dis = [[],[],[],[],[],[],[],[],[],[]]
    for client in clients:
        train_loader = client.train_loader
        distribution = show_distribution(train_loader, args)
        label = np.argmax(distribution)
        client_class.append(label)
        client_class_dis[label].append(client.id)
    print(client_class_dis)
    return client_class_dis

def get_edge_class(args, edges, clients):
    edge_class = [[], [], [], [], []]
    for (i,edge) in enumerate(edges):
        for cid in edge.cids:
            client = clients[cid]
            train_loader = client.train_loader
            distribution = show_distribution(train_loader, args)
            label = np.argmax(distribution)
            edge_class[i].append(label)
    print(f'class distribution among edge {edge_class}')

def initialize_edges_iid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 10 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        for label in range(10):
        #     0-9 labels in total
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace = False)
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
        edges.append(Edge(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
        
    #And the last one, eid == num_edges -1
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients

def initialize_edges_niid(num_edges, clients, args, client_class_dis):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 5 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:`
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    # eid对应label范围
    label_ranges = [[0,1,2,3,4],[1,2,3,4,5],[5,6,7,8,9],[6,7,8,9,0]]
    
    # TODO 修改该函数为均分客户端到各个edge范围内
    # 创建数组存每个edge应该放多少个clients
    num_each_edge_list = [args.num_clients//args.num_edges] * args.num_edges
    # 最后一个 edge放多余的client, 即无法平分数量的 client
    num_each_edge_list[args.num_edges - 1] += (args.num_clients % args.num_edges) 
    
    
    # 对 num_edges - 1个 edge执行
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        # 用提前分组好的标签分布，配给当前edge
        label_range = label_ranges[eid]
        # 进行两次
        
        # 选出对应数量的client
        for i in range(num_each_edge_list[eid]):
            # 在eid对应label范围内随机选择label， 如果对应Label的客户端数组为空，则重新选
            # 一般情况下不会出现client不够用的情况，因此这里没有进行额外处理
            while (True):
                # 取余3防止超越index
                label = np.random.choice(label_ranges[eid % 4], 1, replace=False)[0]
                if len(client_class_dis[label]) > 0:
                    break
                else:
                    # 若不命中则从隔壁借
                    label = np.random.choice(label_ranges[(eid+1) % 4 ], 1, replace=False)[0]
                    if len(client_class_dis[label]) > 0:
                        break
                      
            # 对随机选取的标签对应的client数组中抽一个client_id
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
            # 在字典中除去被抽取客户端的 id
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
            
            # 将抽取的客户端加入到注册列表
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
                    
        edges.append(Edge(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                          ccmab = CCMAB(num_rounds=args.num_communication, id = i, num_context_features = 2)
                          ))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
        
    #And the last one, eid == num_edges -1
    #Find the last available labels
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers),
                      ccmab = CCMAB(num_rounds=args.num_communication, id = i, num_context_features = 2)
                      ))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients

def all_clients_test(server, clients, cids, device):
    [server.send_to_client(clients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_client(clients[cid])
        # The following sentence!
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge

def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all

def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        elif args.model == 'logistic':
            global_nn = LogisticRegression(input_dim=1, output_dim=10)
        else: raise ValueError(f"Model{args.model} not implemented for mnist")
    elif args.dataset == 'cifar10':
        if args.model == 'cnn_complex':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        elif args.model == 'resnet18':
            global_nn = ResNet18()
        else: raise ValueError(f"Model{args.model} not implemented for cifar")
    else: raise ValueError(f"Dataset {args.dataset} Not implemented")
    return global_nn

def create_clients_statues(num_clients, train_loaders, test_loaders, args, device, clients):
    # 定义客户端信道状态Cdt的分布
    cdt_distribution = np.random.uniform(0,1,size=num_clients)
    # 定义客户端离ES的距离dist的分布, 0.04防止计算g得负数
    dist_distribution = np.random.uniform(low=0.04, high=2, size=num_clients)
    # 定义客户端的资源定价charge的分布 (论文对应Pricing funciton)
    charge_distribution = np.random.uniform(low=0.5, high=2, size=num_clients)
     # 定义客户端的计算资源y的分布
    y_distribution = np.random.uniform(low=2, high=4, size=num_clients)
     # 定义客户端带宽的分布
    bandwidth_distribution = np.random.uniform(low=0.3, high=1, size=num_clients)
   
    for i in range(num_clients):
        clients.append(
            Client(
                id=i,
                train_loader=train_loaders[i],
                test_loader=test_loaders[i],
                args=args,
                device=device,
                # Cdt=cdt_distribution,
                dist=dist_distribution[i],
                charge=charge_distribution[i],
                Ynt=y_distribution[i],
                bandwidth=bandwidth_distribution[i]
            )
        )

def update_clients_statue(edges, clients, p_refresh):

    
    # TODO: update clients statue (后续根据数据集要改参数)
    for client in clients:
        p = random.random()
        if p < p_refresh:
            # 选择某部分刷新距离
            client.dist = np.random.uniform(low=0.04, high=2)
            client.Ynt = np.random.uniform(low=2, high=4)
            client.bandwidth = np.random.uniform(low=0.3, high=1)
            client.refresh_Cdt()
            client.refresh_context()

    # TODO: switch cids between different edge (要考虑刷新注册到edge的客户端的列表的问题)

    # pass

def remove_clients_cannot_reach(clients,selected_clients_id_list):
    temp_list = copy.deepcopy(selected_clients_id_list)
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
                    temp_list.remove(id)
   
    # 返回比率，作为CUCB的reward 
    return len(temp_list) / len(selected_clients_id_list) , temp_list



def HierFAVG(args):
    #make experiments repeatable
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    FILEOUT = f"{args.alg}_policy{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"_model_{args.model}iid{args.iid}edgeiid{args.edgeiid}epoch{args.num_communication}" \
              f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}" \
              f"lr_decay_epoch{args.lr_decay_epoch}momentum{args.momentum}"
    writer = SummaryWriter(comment=FILEOUT)
    # Build dataloaders
    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataloaders(args)
    if args.show_dis:
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(distribution)

        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            print(len(test_loader.dataset))
            distribution = show_distribution(test_loader, args)
            print("test dataloader {} distribution".format(i))
            print(f"test dataloader size {test_size}")
            print(distribution)

    # initialize clients and server
    # !初始化客户端状态 
    clients = []
    # COCS状态初始化 + HierFL数据集初始化
    create_clients_statues(args.num_clients, train_loaders, test_loaders, args, device, clients)

    initilize_parameters = list(clients[0].model.shared_layers.parameters())
    nc = len(initilize_parameters)
    for client in clients:
        user_parameters = list(client.model.shared_layers.parameters())
        for i in range(nc):
            user_parameters[i].data[:] = initilize_parameters[i].data[:]

    # Initialize edge server and assign clients to the edge server
    edges = []
    cids = np.arange(args.num_clients)
    clients_per_edge = int(args.num_clients / args.num_edges)
    p_clients = [0.0] * args.num_edges

    if args.iid == -2:
        if args.edgeiid == 1:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_iid(num_edges=args.num_edges,
                                                    clients=clients,
                                                    args=args,
                                                    client_class_dis=client_class_dis)
        elif args.edgeiid == 0:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_niid(num_edges=args.num_edges,
                                                     clients=clients,
                                                     args=args,
                                                     client_class_dis=client_class_dis)
    else:
        # This is randomly assign the clients to edges
        for i in range(args.num_edges):
            # 初始化edge


            #Randomly select clients and assign them
            #TODO 此处提前根据clients、edge数量分割 clients并在edge注册************************

            selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            cids = list (set(cids) - set(selected_cids))
            edges.append(Edge(id = i,
                              cids = selected_cids,
                              shared_layers = copy.deepcopy(clients[0].model.shared_layers),
                            #   初始化每个 edge的 CCMAB对象
                              ccmab = CCMAB(num_rounds=args.num_communication, 
                              id = i, 
                              num_context_features = 2)
                              ))
            # 这里注册Client，会把edge中的sample_registration字典给同步即cid:sample
            [edges[i].client_register(clients[cid]) for cid in selected_cids]

            # TODO:********************************************************
            # 计算范围内所有客户端的样本数总和
            edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
            # 从这里看，p大小取决于样本数量大小，样本多的客户端被选择的概率大
            p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                    list(edges[i].sample_registration.values())]
            # 把edgeServer的receive_buffer、注册的用户id数组、sample_registration给清空
            edges[i].refresh_edgeserver()
            

    # Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.shared_layers))
    # First the clients report to the edge server their training samples
    [cloud.edge_register(edge=edge) for edge in edges]
    p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
                list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()

    #New an NN model for testing error
    global_nn = initialize_global_nn(args)
    if args.cuda:
        global_nn = global_nn.cuda(device)

    # COCS 使用的计数器(Cloud持有，全局字典,{(Client_id,edge_id,(context):number,...})
    counter = {}
    # 画图数组
    all_counter = np.array([0]* args.num_communication * args.num_edge_aggregation)

    #Begin training
    for num_comm in tqdm(range(args.num_communication)):
        cloud.refresh_cloudserver()
        
        [cloud.edge_register(edge=edge) for edge in edges]
        for num_edgeagg in range(args.num_edge_aggregation):
            edge_loss = [0.0]* args.num_edges
            edge_sample = [0]* args.num_edges
            # COCS 用
            edge_num_clients = [0]* args.num_edges            
            correct_all = 0.0
            total_all = 0.0

            # no edge selection included here
            # for each edge, iterate
            # COCS 画图计数器
            len_counter = 0
            for i,edge in enumerate(edges):
                edge.refresh_edgeserver()
                client_loss = 0.0
                selected_cnum = max(int(clients_per_edge * args.frac),1)
                # 存每一个edge所选择的客户端id
                selected_clients_id_list = []
                # 存每一个edge选择后， 能够如期到达的客户端的id
                reached_clients_id_list = []

                # TODO ************************************************ 客户端选择操作处
                # # * Random select
                # selected_cids = np.random.choice(edge.cids,
                #                                  selected_cnum,
                #                                  replace = False,
                #                                  p = p_clients[i])
                # for selected_cid in selected_cids:
                #     edge.client_register(clients[selected_cid])
                
                # * COCS Policy select 
                # 选出该 dge对应的 client对象数组
                if args.alg == 'COCS' or args.alg == 'Random':
                    edge_selected_clients = []
                    for client in clients:
                        if client.id in edge.cids:
                            edge_selected_clients.append(client)
                    # 初始化CCMAB
                    edge.CCMAB.get_available_clients(edge_selected_clients)
                    # 进行COCS选择
                    _, selected_clients_id_list = edge.CCMAB.get_selected_clients(counter, num_comm * args.num_edge_aggregation + num_edgeagg +1, args.alg )
                    # 根据Client实际条件，将不能按时到达的client去除
                    _, reached_clients_id_list = remove_clients_cannot_reach(clients,selected_clients_id_list)
                    # 更新counter
                    edge.CCMAB.update(reached_clients_id_list,counter)
                    # 更新selected_cid
                    selected_cids = reached_clients_id_list
                    # edge.cids = selected_cids
                    # 重新注册clients
                    [edge.client_register(clients[cid]) for cid in selected_cids]
                    # 更新画图 counter
                    len_counter = len_counter + len(reached_clients_id_list)
                    print("reached :",reached_clients_id_list)
                    
                # * CUCB Policy select 
                elif args.alg == 'CUCB':
                    edge_selected_clients = []
                    for client in clients:
                        if client.id in edge.cids:
                            # id 列表
                            edge_selected_clients.append(client.id)
                    
                    # 检查edge是否已经初始化 
                    for edge in edges:
                        if isinstance(edge.CUCB, CombinatorialUCB):
                            pass
                        else:
                            # 先随意初始化
                            edge.CUCB = CombinatorialUCB(1,1)
                            cost_list = []
                            for id in edge.cids:
                                for client in clients:
                                    if id == client.id:
                                        cost_list.append(client.Ynt * client.charge / 8 )
                            cost_list = [round(x, 2) for x in cost_list]
                            print(cost_list)
                            combinations = edge.CUCB.find_all_budget_combinations(tuple(edge_selected_clients), tuple(cost_list), target_budget=6)
                            num_arms = num_combinations = len(combinations)
                            edge.CUCB = CombinatorialUCB(num_arms,1)
                    
                    # 去重
                    # combinations = set(map(frozenset, combinations))
                    # 计数
                    selected_arm = edge.CUCB.select_arm()
                    # 在选择的臂上选择一个动作（这里简化为直接使用臂的索引作为动作）
                    selected_action = 1
                    # 模拟观察到的奖励（可以根据具体情境更改此部分）
                    reward, reached_clients_id_list = remove_clients_cannot_reach(clients,copy.deepcopy(combinations[selected_arm]))
                    # 抽出所选择的客户端组合(深拷贝)
                    selected_clients_id_list = reached_clients_id_list

                    # 更新状态
                    edge.CUCB.update(selected_arm, selected_action, reward)
                    
                    len_counter = len_counter + len(selected_clients_id_list)
                    print("reached:", reached_clients_id_list)
                    
                    selected_cids = selected_clients_id_list
                    # edge.cids = selected_cids
                    # 重新注册clients
                    [edge.client_register(clients[cid]) for cid in selected_cids]
                    

                # TODO ************************************************
                # 发送边缘服务器的模型参数给选中的客户端
                for selected_cid in selected_cids:
                    edge.send_to_client(clients[selected_cid])
                    clients[selected_cid].sync_with_edgeserver()
                    client_loss += clients[selected_cid].local_update(num_iter=args.num_local_update,
                                                                      device = device)
                    clients[selected_cid].send_to_edgeserver(edge)
                edge_loss[i] = client_loss
                edge_sample[i] = sum(edge.sample_registration.values())
                # COCS用
                edge_num_clients[i] = len(selected_clients_id_list)

                edge.aggregate(args)
                # HFL用
                # correct, total = all_clients_test(edge, clients, edge.cids, device)
                # COCS用
                correct, total = all_clients_test(edge, clients, selected_cids, device)
                correct_all += correct
                total_all += total
            # end interation in edges HFL用
            # all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            
            # COCS loss
            all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            all_loss = 0
            for i in range(len(edge_loss)):
                avg_edge_i_loss = edge_loss[i] / edge_num_clients[i]
                all_loss += avg_edge_i_loss
            all_loss = all_loss / args.num_edges
                
                
            
            avg_acc = correct_all / total_all
            
            writer.add_scalar(f'Partial_Avg_Train_loss',
                          all_loss,
                          num_comm* args.num_edge_aggregation + num_edgeagg + 1)
            writer.add_scalar(f'All_Avg_Test_Acc_edgeagg',
                          avg_acc,
                          num_comm * args.num_edge_aggregation + num_edgeagg + 1)
            
            # 
            all_counter[num_comm * args.num_edge_aggregation + num_edgeagg] = len_counter
            # TODO考虑客户端移动到别的ES区域的情况
            update_clients_statue(edges, clients, 0.1)

        # Now begin the cloud aggregation
        for edge in edges:
            edge.send_to_cloudserver(cloud)
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        global_nn.load_state_dict(state_dict = copy.deepcopy(cloud.shared_state_dict))
        global_nn.train(False)
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_nn, device)
        avg_acc_v = correct_all_v / total_all_v
        writer.add_scalar(f'All_Avg_Test_Acc_cloudagg_Vtest',
                          avg_acc_v,
                          num_comm + 1)

    writer.close()
    print(f"The final virtual acc is {avg_acc_v}")

    # COCS画图
    time_list = list(range(1,num_comm * args.num_edge_aggregation + num_edgeagg + 1))
    plt.plot(all_counter, label=args.alg)
    plt.xlabel("round")
    plt.ylabel("number of clients")
    plt.legend()
    plt.show()
    
    np.savetxt(args.alg + '-result.csv', all_counter, delimiter=',')

def main():
    args = args_parser()
    HierFAVG(args)

if __name__ == '__main__':
    main()