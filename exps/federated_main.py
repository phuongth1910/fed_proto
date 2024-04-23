#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist, MLP
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
#vòng lặp này đào tạo toàn cục 
    for round in tqdm(range(args.rounds)): #tqdm để tạo thanh tiến trình theo dõi quá trình đào tạo 
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users: #vòng lặp qua từng ng dùng 
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx]) # tạo mô hình cục bộ cho ng dùng thứ idx
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round) #cập nhật trọng số dựa trên global_protos và mô hình cục bộ hiện tại 
            #tổng hợp và biểu diễn thông tin trên tensorbroad
            agg_protos = agg_func(protos) #tổng hợp mô hình mẫu 
            #cập nhật lại danh sách local_weight,loss,protos
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos
            #ghi thông tin vào tensorbroad sau đó tính loss,acc của từng ng dùng 
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights #sao chép danh sách trọng số cục bộ chú ý gán tham chiếu, sự thay đổi ở list ảnh hưởng đến local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx]) #sao chép mô hình cục bộ của người dùng hiện tại idx từ list mô hình local
            local_model.load_state_dict(local_weights_list[idx], strict=True) #cập nhật trạng thái mô hình với trọng số mới trong list
            local_model_list[idx] = local_model #mô hình cục bộ mới được cập nhật vào list

        # update global protos
        global_protos = proto_aggregation(local_protos) #tổng hợp lại 

        loss_avg = sum(local_losses) / len(local_losses)  # để đo lường hiệu suất
        train_loss.append(loss_avg)
        
    #kiểm thử trên tập dữ liệu test, sd model local với 3 đầu ra là acc_list_g/l, loss list     
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # save protos
    if args.dataset == 'mnist': #ktra dataset
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            #tong hop va bieu dien tensorboard
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

if __name__ == '__main__':
    start_time = time.time() #khởi tạo thgian, ghi lai thời điểm bdau 

    args = args_parser() #arg_parser để phân tích và trả về tham số 
    exp_details(args) #in ra thông tin về tham số 

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu' #ktra gpu 
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu) #đặt gpu cụ thể được chỉ định trong args.gpu
        torch.cuda.manual_seed(args.seed) #hạt giống ngẫu nhiên cho cuda
        torch.manual_seed(args.seed) #hạt giống ngẫu nhiên cho pytorch
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed) #đặt hạt giống ngẫu nhiên cho nympy
    random.seed(args.seed) #thư viện ngẫu nhiên 

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users) #mảng chứa số lượng lớp mà mỗi người dùng được gán
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users) #mảng chứa slg hình ảnh đào tạo mà mỗi ng dùng có, tạo ngẫu nhiên trên dataset
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'mri':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list) #getdata là hàm tạo và tải tập dữ liệu dựa trên tham số args,n_list (slg lớp mỗi ng dùng),k_list(slg hình ảnh đào tạo mỗi ng dùng)

    # Build models
    local_model_list = [] #ds mô hình cục bộ cho mỗi ng dùng 
    for i in range(args.num_users): #duyệt qua mỗi ng dùng 
        if args.dataset == 'mnist':
            if args.mode == 'model_heter': #xác định xem là model_heter hay task_heter
                if i<7: #i là chỉ số ng dùng 
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args) #tạo đtg mô hình CNN

        elif args.dataset == 'mri':
            if args.mode == 'model_heter': #xác định xem là model_heter hay task_heter
                if i<7: #i là chỉ số ng dùng 
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = MLP(args=args) #tạo đtg mô hình CNN
        
        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i<10:
                    args.stride = [1,4]
                else:
                    args.stride = [2,2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes) #tạo 1 dtg mô hình resnet18 dựa vào tham số args., mô hình có lớp đầu ra là args.num_classes
            initial_weight = model_zoo.load_url(model_urls['resnet18']) #tải trọng số được đào tạo trước của mô hình resnet18 từ thư viện pytorch
            local_model = resnet #vừa gán vừa tạp biến 
            initial_weight_1 = local_model.state_dict() #lấy trọng số mô hình vừa tạo 
            for key in initial_weight.keys(): 
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1': #ktra xem khóa đó có phải fully connceted(fc), của lớp convolutional đầu tiên (conv1) hoặc của lớp batch normalization đầu tiên (bn1)
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight) #áp dụng trọng số đã điều chỉnh vào mô hình resnet18

        local_model.to(args.device) #chuyển đtg mô hình lên thiết bị đc chỉ đỉnh bởi args.device
        local_model.train() 
        local_model_list.append(local_model) #thêm đtg mô hình đã đào tạo vào list

    if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)