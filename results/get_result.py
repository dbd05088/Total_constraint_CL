import os
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')
dir = 'cifar100'

def print_from_log(exp_name, seeds=(1, 2, 3)):
    A_auc = []
    A_last = []
    A_online = []
    F_last = []
    IF_avg = []
    KG_avg = []
    FLOPS = []
    for i in seeds:
        f = open(f'{exp_name}/seed_{i}.log', 'r')
        lines = f.readlines()

        for line in lines:
            if 'gdumb' in exp_name:
                if 'Test' in line:
                    list = line.split(' ')
                    a_last = float(list[13])*100
            if 'A_auc' in line:
                list = line.split(' ')
                A_auc.append(float(list[4])*100)
                #A_online.append(float(list[7])*100)
                if 'gdumb' in exp_name:
                    A_last.append(a_last)
                else:
                    A_last.append(float(list[7])*100)
                FLOPS.append(float(list[-1])/100)
                # IF_avg.append(float(list[16]) * 100)
                # KG_avg.append(float(list[19]) * 100)
                break
        #kl = np.load(f'{exp_name}/seed_{i}_forgetting.npy')
        #kg = np.load(f'{exp_name}/seed_{i}_knowledge_gain.npy')
        #tk = np.load(f'{exp_name}/seed_{i}_total_knowledge.npy')
        # tk = np.insert(tk, 0, 0)
        #klrate = np.clip(kl[2:]/tk[2:], 0, 1)       # print(klrate*100)
        #kgrate = kg[1:]/(1-tk[1:])
        #IF_avg.append(klrate[1:].mean()*100)
        #KG_avg.append(kgrate[1:].mean()*100)
    if np.isnan(np.mean(A_auc)):
        pass
    else:
        print(f'Exp:{exp_name} \t\t\t {np.mean(A_auc):.2f}/{np.std(A_auc):.2f} \t {np.mean(A_last):.2f}/{np.std(A_last):.2f} \t  {np.mean(IF_avg):.2f}/{np.std(IF_avg):.2f}  \t  {np.mean(KG_avg):.2f}/{np.std(KG_avg):.2f}  \t  {np.mean(FLOPS):.2f}/{np.std(FLOPS):.2f}|')

print("A_auc, A_last, IF_avg, KG_avg FLOPS")

exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])

for exp in exp_list:
    try:
        print_from_log(exp)
    except:
        pass





