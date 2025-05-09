import os
import warnings
import argparse
import wandb

wandb.require("core")

os.environ['TORCH_CUDA_ARCH_LIST']  = '8.6'

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*TORCH_CUDA_ARCH_LIST.*')

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../shape_assembly'))
from config import get_cfg_defaults
from datasets.dataloader.dataloader_B import OurDataset
from models.train.network_vnn_B import ShapeAssemblyNet_B_vnn
from models.train.network_vnn_A import ShapeAssemblyNet_A_vnn
import utils
from tqdm import tqdm

orange = '\033[38;5;214m'
reset = '\033[0m'

def setup(rank, world_size, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '52997'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

    
def train(rank, world_size, conf):
    # create the log_file
    os.makedirs(conf.exp.log_dir, exist_ok=True)

    # create the vis file
    os.makedirs(conf.exp.vis_dir, exist_ok=True)

    setup(rank, world_size, conf)

    if dist.get_rank() == 0:
        wandb.init(project='shape-matching', notes='weak baseline', config=conf)
        wandb.define_metric("test/epoch/*", step_metric="test/epoch/epoch")
        # wandb.define_metric("val/step/*", step_metric="val/step/step")
        wandb.define_metric("train/network_B/*", step_metric="train/network_B/step")
        wandb.define_metric("train/network_A/*", step_metric="train/network_A/step")

    data_features = ['src_pc', 'src_rot', 'src_trans', 'tgt_pc', 'tgt_rot', 'tgt_trans', 'partA_symmetry_type', 'partB_symmetry_type','predicted_partB_rotation', 'predicted_partB_position', 'predicted_partA_rotation', 'predicted_partA_position'] 

    network_B = ShapeAssemblyNet_B_vnn(cfg=conf, data_features=data_features)
    network_B.cuda(rank)

    # Load pretrained weights for network_B
    network_B.load_state_dict(torch.load(os.path.join(conf.exp.log_dir, 'ckpts', 'network_B.pth')))
 
    for param in network_B.parameters():
        param.requires_grad = False


    network_A = ShapeAssemblyNet_A_vnn(cfg=conf, data_features=data_features)
    network_A.cuda(rank)

    # Load pretrained weights for network_A
    network_A.load_state_dict(torch.load(os.path.join(conf.exp.log_dir, 'ckpts', 'network_A.pth')))
    network_A = DDP(network_A, device_ids=[rank])

    for param in network_A.parameters():
        param.requires_grad= False

    # Initialize train dataloader
    train_data = OurDataset(
        data_root_dir=conf.data.root_dir,
        data_csv_file=conf.data.train_csv_file,
        data_features=data_features,
        num_points=conf.data.num_pc_points
    )
    train_data.load_data()

    print('Len of Train Data: ', len(train_data))

    # Initialize val dataloader
    val_data = OurDataset(
        data_root_dir=conf.data.root_dir,
        data_csv_file=conf.data.val_csv_file,
        data_features=data_features,
        num_points=conf.data.num_pc_points
    )
    val_data.load_data()

    print('Len of Val Data: ', len(val_data))

    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=conf.exp.batch_size,
        num_workers=conf.exp.num_workers,
        pin_memory=True,
        # shuffle=True,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler
    )

    network_opt = torch.optim.Adam(list(network_A.parameters()), lr=conf.optimizer.lr, weight_decay=conf.optimizer.weight_decay)
    val_num_batch = len(val_dataloader)
    print(f"\033[33mNumber of batches in val_dataloader: {len(val_dataloader)}\033[0m")
    print(f"\033[33mNumber of samples in val_data: {len(val_data)}\033[0m")

    # === Evaluation ===
    tot, tot_gd, tot_r, tot_t, tot_CD_A, tot_CD_B = 0, 0, 0, 0, 0, 0

    network_A.train()
    network_B.train()

    for val_batch_ind, val_batch in enumerate(val_dataloader):
        for key in val_batch.keys():
            if key not in ['category_name', 'cut_name', 'shape_id', 'result_id']:
                val_batch[key] = val_batch[key].to(rank)
        with torch.no_grad():
            partB_eval_metric, partB_pos, partB_rot, *_ = network_B.forward_pass(
                batch_data=val_batch, device=rank, mode='val', vis_idx=val_batch_ind)
            val_batch['predicted_partB_position'] = partB_pos
            val_batch['predicted_partB_rotation'] = partB_rot

            partA_eval_metric, *_ = network_A.module.forward_pass(
                batch_data=val_batch, device=rank, mode='val', vis_idx=val_batch_ind)

            GD_B, R_B, T_B, _, _, CD_B = partB_eval_metric
            GD_A, R_A, T_A, _, _, CD_A = partA_eval_metric

        tot += 1
        tot_gd += (GD_A.mean() + GD_B.mean()) / 2
        tot_r += (R_A.mean() + R_B.mean()) / 2
        tot_t += (T_A.mean() + T_B.mean()) / 2
        tot_CD_A += CD_A.mean()
        tot_CD_B += CD_B.mean()

    if dist.get_rank() == 0:
        print(f"\033[1;32m[Eval Results]\033[0m")
        print(f"avg_gd: {tot_gd / tot}")
        print(f"avg_r: {tot_r / tot}")
        print(f"avg_t: {tot_t / tot}")
        print(f"avg_CD_A: {tot_CD_A / tot}")
        print(f"avg_CD_B: {tot_CD_B / tot}")

        wandb.log({
            'test/epoch/avg_gd': (tot_gd / tot).item(),
            'test/epoch/avg_r': (tot_r / tot).item(),
            'test/epoch/avg_t': (tot_t / tot).item(),
            'test/epoch/avg_CD_A': (tot_CD_A / tot).item(),
            'test/epoch/avg_CD_B': (tot_CD_B / tot).item(),
            'test/epoch/epoch': 0
        })

    dist.barrier()
    cleanup()


def main(cfg):
    world_size = len(cfg.gpus)

    mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == '__main__':

    wandb.login()

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg_file', default='', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()
    args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    cfg.gpus = cfg.exp.gpus

    cfg.freeze()
    print(cfg)

    main(cfg)