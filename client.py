# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
from torch.autograd import Variable
import torch
from models.initialize_model import initialize_model
import copy
import numpy as np

class Client():

    def __init__(self, id, train_loader, test_loader, args, device , Ynt=0, dist=0, charge=0, bandwidth=0):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        #record local update epoch
        self.epoch = 0
        # record the time
        self.clock = []


        # COCS策略模拟参数
        # 客户端计算资源
        self.Ynt = Ynt
        # 客户端离ES距离
        self.dist = dist
        # 资源定价
        self.charge = charge
        # 客户端带宽
        self.bandwidth = bandwidth
        # 客户端信道状态
        self.Cdt = 0
        # 上下文
        self.context = np.array([self.Cdt])

        self.refresh_Cdt()
        self.refresh_context()

    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels)
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end: break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch = self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
                                        )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None

    # COCS策略使用
    def refresh_context(self):
        context = self.context = np.array([self.Cdt])
        return context

    # COCS策略使用
    def refresh_Cdt(self):
        gDt = (37.6) * np.log(self.dist) + 128.1
        # 噪声功率
        No = 12000
        # No = random.randint(10000, 14000)
        Cdt = self.Cdt = np.log2(1 + gDt * 23 / No)
        return Cdt