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
import utils
# from tensorboardX import SummaryWriter
from tqdm import tqdm

orange = '\033[38;5;214m'
reset = '\033[0m'

def setup(rank, world_size, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12719'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    
def train(rank, world_size, conf):
    os.makedirs(conf.exp.log_dir, exist_ok=True)
    os.makedirs(conf.exp.vis_dir, exist_ok=True)
    setup(rank, world_size, conf)

    if dist.get_rank() == 0:
        wandb.init(project='shape-matching', notes='weak baseline', config=conf)
        wandb.define_metric("test/epoch/*", step_metric="test/epoch/epoch")
        wandb.define_metric("train/network_B/*", step_metric="train/network_B/step")

    data_features = ['src_pc', 'src_rot', 'src_trans', 'tgt_pc', 'tgt_rot', 'tgt_trans', 'partA_symmetry_type', 'partB_symmetry_type','predicted_partB_rotation', 'predicted_partB_position', 'predicted_partA_rotation', 'predicted_partA_position'] 

    network_B = ShapeAssemblyNet_B_vnn(cfg=conf, data_features=data_features)
    network_B.cuda(rank)
    network_B = DDP(network_B, device_ids=[rank])

    # Initialize train dataloader
    train_data = OurDataset(
        data_root_dir=conf.data.root_dir,
        data_csv_file=conf.data.train_csv_file,
        data_features=data_features,
        num_points=conf.data.num_pc_points
    )
    train_data.load_data()

    print('Len of Train Data: ', len(train_data))

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=conf.exp.batch_size,
        num_workers=conf.exp.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        sampler=train_sampler
    )

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
        shuffle=False,
        drop_last=False,
        sampler=val_sampler
    )

    network_opt = torch.optim.Adam(list(network_B.parameters()), lr=conf.optimizer.lr, weight_decay=conf.optimizer.weight_decay)
    val_num_batch = len(val_dataloader)
    print(f"\033[33mNumber of batches in val_dataloader: {len(val_dataloader)}\033[0m")
    print(f"\033[33mNumber of samples in val_data: {len(val_data)}\033[0m")
    train_num_batch = len(train_dataloader)

    val_step = 0
    for epoch in tqdm(range(1, conf.exp.num_epochs + 1)):
        print("\033[1;32m", "Epoch {} In training".format(epoch), "\033[0m")
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)
        val_fraction_done = 0.0
        val_batch_ind = -1

        # train for every batch
        for train_batch_ind, batch in train_batches:
            if train_batch_ind % 50 == 0 and dist.get_rank() == 0:
                print("*" * 10)
                print(epoch, train_batch_ind)
                # print("*" * 10)
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = True

            # save checkpoint of network_A and network_B
            if epoch % 10 == 0 and train_batch_ind == 0 and dist.get_rank() == 0:
                with torch.no_grad():

                    os.makedirs(os.path.join(conf.exp.log_dir, 'ckpts'), exist_ok=True)
                    torch.save(network_B.module.state_dict(), os.path.join(conf.exp.log_dir, 'ckpts', '%d-network_B.pth' % epoch), _use_new_zipfile_serialization=False)
               
            network_B.train()
            for key in batch.keys():
                if key not in ['category_name', 'cut_name', 'shape_id', 'result_id']:
                    batch[key] = batch[key].cuda(non_blocking=True)

            partB_predictions = network_B.module.training_step(batch_data=batch, device=rank, batch_idx=train_batch_ind)
            total_loss_B = partB_predictions["total_loss"]
            rot_loss_B = partB_predictions["rot_loss"]
            trans_loss_B = partB_predictions["trans_loss"]

            batch['predicted_partB_position'] = partB_predictions['predicted_partB_position']
            batch['predicted_partB_rotation'] = partB_predictions['predicted_partB_rotation']

            if train_batch_ind % 50 == 0 and dist.get_rank() == 0:
                print(total_loss_B.detach().cpu().numpy())

            step = train_step
            if dist.get_rank() == 0:
                wandb.log({'train/network_B/total_loss': total_loss_B.item(), 'train/network_B/rot_loss': rot_loss_B.item(), 'train/network_B/trans_loss': trans_loss_B.item(), 'train/network_B/step':step})
            
            step = train_step
        
            total_loss = total_loss_B

            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            
        dist.barrier()   
            
        if epoch % 1 == 0:
            tot = 0
            tot_gd_A = 0
            tot_gd_B = 0
            tot_r_A = 0
            tot_r_B = 0
            tot_t_A = 0
            tot_t_B = 0
            tot_pa_A = 0
            tot_pa_B = 0
            tot_pa = 0
            tot_t = 0
            tot_r = 0
            tot_gd = 0
            tot_pa_threshold = 0
            tot_CD_A = 0
            tot_CD_B = 0
            tot_pa_threshold_A = 0
            tot_pa_threshold_B = 0
            val_batches = enumerate(val_dataloader, 0)
            val_fraction_done = 0.0
            val_batch_ind = -1

            total_loss_epoch = 0
            rot_loss_epoch = 0
            trans_loss_epoch = 0

            for val_batch_ind, val_batch in val_batches:
                if val_batch_ind % 50 == 0:
                    print("*" * 10)
                    print(epoch, val_batch_ind)
                    print("*" * 10)

                # network_A.train()
                network_B.train()

                for key in val_batch.keys():
                    if key not in ['category_name', 'cut_name', 'shape_id', 'result_id']:
                        val_batch[key] = val_batch[key].to(rank)
                with torch.no_grad():
        
                    partB_eval_metric, partB_position, partB_rotation, partB_total_loss, partB_point_loss, partB_rot_loss, partB_trans_loss, partB_recon_loss = network_B.module.forward_pass(batch_data=val_batch, device=rank, mode='val', vis_idx=val_batch_ind)
                    GD_B, R_error_B, RMSE_T_B, PA_threshold_B, PA_B, CD_B = partB_eval_metric

                    total_loss_epoch += partB_total_loss.item()
                    rot_loss_epoch += partB_rot_loss.item()
                    trans_loss_epoch += partB_trans_loss.item() 

                    val_batch['predicted_partB_position'] = partB_position
                    val_batch['predicted_partB_rotation'] = partB_rotation
                
                val_step += 1

                # [todo]

                # tot_gd_A += GD_A.mean()
                tot_gd_B += GD_B.mean()
                # tot_r_A += R_error_A.mean()   # the rotation loss for each batch
                tot_r_B += R_error_B.mean()   # the rotation loss for each batch
                # tot_t_A += RMSE_T_A.mean()
                tot_t_B += RMSE_T_B.mean()
                # tot_pa_threshold_A += PA_threshold_A.mean()
                tot_pa_threshold_B += PA_threshold_B.mean()
                # tot_pa_A += PA_A.mean()
                tot_pa_B += PA_B.mean()
                # tot_CD_A += CD_A.mean()
                tot_CD_B += CD_B.mean()
                
                tot_r =  tot_r_B
                tot_t =  tot_t_B
                tot_gd = (tot_gd_A + tot_gd_B) / 2
                tot += 1
            
            if dist.get_rank() == 0:
                # print("logging wandb !!! in val !!!!! ")
                print("\033[1;32m", "Epoch {} In validation".format(epoch), "\033[0m")

                print("avg_gd: ", tot_gd / tot)
                print("avg_r: ", tot_r / tot)
                print("avg_t: ", tot_t / tot)
                print("avg_CD_1: ", tot_CD_A / tot)
                print("avg_CD_2: ", tot_CD_B / tot)

                log_data = {
                    'test/epoch/avg_gd':(tot_gd/tot).item(),
                    'test/epoch/avg_r': (tot_r/tot).item(),
                    'test/epoch/avg_t': (tot_t/tot).item(),
                    'test/epoch/avg_CD_B': (tot_CD_B/tot).item(),
                    'test/epoch/total_loss': total_loss_epoch,
                    'test/epoch/rot_loss': rot_loss_epoch,
                    'test/epoch/trans_loss': trans_loss_epoch,
                    'test/epoch/epoch': epoch
                }
                wandb.log(
                log_data
                )
            
            # val_step += 1
        
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