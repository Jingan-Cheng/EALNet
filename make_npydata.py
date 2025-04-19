import os
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Dataset")
parser.add_argument("--root_dir", type=str, default="/scratch/FIDTM")
args = parser.parse_args()

if not os.path.exists('.npydata/npy'):
    os.makedirs('.npydata/npy')

'''please set your dataset path'''
shanghai_root = os.path.join(args.root_dir, "ShanghaiTech")
jhu_root = os.path.join(args.root_dir, "jhu_crowd_v2.0")
qnrf_root = os.path.join(args.root_dir, "UCF-QNRF_ECCV18")

select_shanghai = "/scratch/FIDTM/adaptation_dataset/select_dataset/A_to_B"
unlabeled_shanghai = "/scratch/FIDTM/adaptation_dataset/unlabeled_dataset/A_to_B"

select_qnrf_root = "/scratch/FIDTM/adaptation_dataset/select_dataset/J_to_Q"
unlabeled_qnrf_root = "/scratch/FIDTM/adaptation_dataset/unlabeled_dataset/J_to_Q"



try:
    shanghaiAtrain_path = select_shanghai + '/part_A_final/train_data/images/'
    shanghaiAtest_path = "/scratch/FIDTM/ShanghaiTech" + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/select_ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/select_ShanghaiA_test.npy', test_list)

    print("generate select ShanghaiA image list successfully")
except:
    print("The select ShanghaiA dataset path is wrong. Please check you path.")

try:
    shanghaiAtrain_path = unlabeled_shanghai + '/part_A_final/train_data/images/'
    shanghaiAtest_path = "/scratch/FIDTM/ShanghaiTech" + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/unlabeled_ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/unlabeled_ShanghaiA_test.npy', test_list)

    print("generate unlabeled ShanghaiA image list successfully")
except:
    print("The unlabeled ShanghaiA dataset path is wrong. Please check you path.")    
    
try:
    shanghaiAtrain_path = shanghai_root + '/part_A_final/train_data/images/'
    shanghaiAtest_path = shanghai_root + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully")
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")


try:
    shanghaiBtrain_path = select_shanghai + '/part_B_final/train_data/images/'
    shanghaiBtest_path = "/scratch/FIDTM/ShanghaiTech" + '/part_B_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/select_ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/select_ShanghaiB_test.npy', test_list)
    print("Generate select ShanghaiB image list successfully")
except:
    print("The select ShanghaiB dataset path is wrong. Please check your path.")
    
try:
    shanghaiBtrain_path = unlabeled_shanghai + '/part_B_final/train_data/images/'
    shanghaiBtest_path = "/scratch/FIDTM/ShanghaiTech" + '/part_B_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/unlabeled_ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/unlabeled_ShanghaiB_test.npy', test_list)
    print("Generate unlabeled ShanghaiB image list successfully")
except:
    print("The unlabeled ShanghaiB dataset path is wrong. Please check your path.")

try:
    shanghaiBtrain_path = shanghai_root + '/part_B_final/train_data/images/'
    shanghaiBtest_path = shanghai_root + '/part_B_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully")
except:
    print("The ShanghaiB dataset path is wrong. Please check your path.")

try:
    Qnrf_train_path = unlabeled_qnrf_root + '/train_data/images/'
    Qnrf_test_path = "/scratch/FIDTM/UCF-QNRF_ECCV18" + '/test_data/images/'

    train_list = []
    for filename in os.listdir(Qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Qnrf_train_path + filename)
    train_list.sort()
    np.save('./npydata/unlabeled_qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(Qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(Qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/unlabeled_qnrf_test.npy', test_list)
    print("Generate unlabeled QNRF image list successfully")
except:
    print("The unlabeled QNRF dataset path is wrong. Please check your path.")

try:
    Qnrf_train_path = select_qnrf_root + '/train_data/images/'
    Qnrf_test_path = "/scratch/FIDTM/UCF-QNRF_ECCV18" + '/test_data/images/'

    train_list = []
    for filename in os.listdir(Qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Qnrf_train_path + filename)
    train_list.sort()
    np.save('./npydata/select_qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(Qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(Qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/select_qnrf_test.npy', test_list)
    print("Generate select QNRF image list successfully")
except:
    print("The select QNRF dataset path is wrong. Please check your path.")

try:
    Qnrf_train_path = qnrf_root + '/train_data/images/'
    Qnrf_test_path = qnrf_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(Qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Qnrf_train_path + filename)
    train_list.sort()
    np.save('./npydata/qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(Qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(Qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/qnrf_test.npy', test_list)
    print("Generate QNRF image list successfully")
except:
    print("The QNRF dataset path is wrong. Please check your path.")

