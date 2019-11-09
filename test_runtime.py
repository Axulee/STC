import argparse
import time
import os
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F

parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
parser.add_argument('--img_feature_dim',type=int, default=224)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default=None)

args = parser.parse_args()

def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

gpu = list(args.gpus)
if len(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

this_weights = args.weights
this_test_segments = args.test_segments
is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
if 'RGB' in this_weights:
    modality = 'RGB'
else:
    modality = 'Flow'
this_arch = this_weights.split('TSM_')[1].split('_')[2]

num_class = 174

net = TSN(num_class, this_test_segments if is_shift else 1, modality,
          base_model=this_arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in this_weights,
          )

if 'tpool' in this_weights:
    from ops.temporal_shift import make_temporal_pool
    make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

checkpoint = torch.load(this_weights)
checkpoint = checkpoint['state_dict']

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }

for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net.load_state_dict(base_dict)


input_size = net.scale_size if args.full_res else net.input_size
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 3:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 5:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))


net = torch.nn.DataParallel(net.cuda())
net.eval()


total_num = 11522
output = []

def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    # if modality == 'RGB':
    #     length = 3
    # elif modality == 'Flow':
    #     length = 10
    # elif modality == 'RGBDiff':
    #     length = 18
    # else:
    #     raise ValueError("Unknown modality "+modality)

    rst = net(data).data.cpu().numpy().copy()

    return i, rst, label


proc_start_time = time.time()

for i in range(11522):
    data = torch.randn(1,3,8,224,224)
    label = torch.tensor([1])
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} video/sec'.format(i, i+1, total_num, (i+1)/float(cnt_time)))

# video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

# video_labels = [x[1] for x in output]


# cf = confusion_matrix(video_labels, video_pred).astype(float)

# cls_cnt = cf.sum(axis=1)
# cls_hit = np.diag(cf)

# cls_acc = cls_hit / cls_cnt

# print(cls_acc)

# print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))


