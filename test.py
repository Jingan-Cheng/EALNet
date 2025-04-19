from __future__ import division
import warnings


import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from misc.image import *
from misc.utils import *

import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

setup_seed(args.seed)


def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_test.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    model = get_model(args)
    
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ])

    logger.info(args['pre'])

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args["pre"]:
        if os.path.isfile(args["pre"]):
            logger.info("=> loading checkpoint '{}'".format(args["pre"]))
            checkpoint = torch.load(args["pre"])
            
            state_dict = {}

            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    new_key = k.replace('module.', '', 1)   
                else:
                    new_key = k
                state_dict[new_key] = v
            
            model.load_state_dict(state_dict)
            args["start_epoch"] = checkpoint["epoch"]
            args["best_pred"] = checkpoint["best_prec1"]
        else:
            logger.info("=> no checkpoint found at '{}'".format(args["pre"]))

    torch.set_num_threads(args['workers'])
    logger.info(f"best_pred: {args['best_pred']} start_epoch: {args['start_epoch']}")

    if args['preload_data'] == True:
        test_data = pre_data(val_list, args, train=False)
    else:
        test_data = val_list
    prec1, visi = validate(test_data, model, args, val_list)

    is_best = prec1 < args['best_pred']
    args['best_pred'] = min(prec1, args['best_pred'])

    logger.info(f"\nThe visualizations are provided in {args['save_path']}")
    save_checkpoint({
        'arch': args['pre'],
        'state_dict': model.state_dict(),
        'best_prec1': args['best_pred'],
        'optimizer': optimizer.state_dict(),
    }, visi, is_best, args['save_path'])


def pre_data(train_list, args, train):
    logger.info("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys

def validate(Pre_data, model, args, val_list):
    logger.info('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    if not os.path.exists('./local_eval/point_files'):
        os.makedirs('./local_eval/point_files')
    f_loc = open("./local_eval/point_files/ShanghaiA_baseline_coat_dense_localization.txt", "w+")

    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):

        count = 0
        img = img.cuda()

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            d6 = model(img)
            count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

            if args['visual'] == True:
                if not os.path.exists(args['save_path'] + '_box/'):
                    os.makedirs(args['save_path'] + '_box/')
                ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
                show_fidt = show_map(d6.data.cpu().numpy())
                gt_show = show_map(fidt_map.data.cpu().numpy())
                res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                cv2.imwrite(args['save_path'] + '_box/' + fname[0], res)

        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 1 == 0:
            logger.info('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
            visi.append(
                [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                 fname])
            index += 1

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    logger.info(f"\n* MAE {mae:.3f}\n* MSE {mse:.3f}")
    
    return mae, visi

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()
    if args['dataset'] == 'UCF_QNRF':
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    else:
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc


def generate_point_map(kpoint, f_loc, rate=1):
    pred_coor = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map

def generate_bounding_boxes(kpoint, fname):
    # change data path
    # change
    Img_data = cv2.imread(
        '/home/wenzhe/bulk/Experiments/Crowd/FIDIM/MANet/ShanghaiTech/part_A_final/test_data/images/' + fname[0])
    ori_Img_data = Img_data.copy()
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                 (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return ori_Img_data, Img_data

def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__': 
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))

    os.makedirs(params["save_path"], exist_ok=True)
       
    params["task_name"] = "_".join(params["save_path"].split("/")[-1:])   
    
    global logger 
    logger = get_logger("Test", f"{params['save_path']}/test.log")
    
    params["save_path"] = os.path.join(params["save_path"], "test") 
    os.makedirs(params["save_path"], exist_ok=True)
    
    logger.debug(tuner_params)
    logger.info(params)

    main(params)
