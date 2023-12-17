import argparse
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import shutil
import random
import sys
from tensorboardX import SummaryWriter

from configs import get_config
from dataloaders.PFERD_dataset import DatasetPFERD
from models.egohmr.hSMALhmr import hSMALDiffusion
from diffusion.model_util import create_gaussian_diffusion
from diffusion.resample import UniformSampler
from utils.other_utils import *
from render.renderer import COLORRenderer
from render.SMAL.smal_torch.smal_torch import SMAL

parser = argparse.ArgumentParser(description='EgoHMR training code')
parser.add_argument('--gpu_id', type=int, default='0')

parser.add_argument('--model_cfg', type=str, default='configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--save_dir', type=str, default='/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/results/test', help='path to save train logs and models')
parser.add_argument('--dataset_root', type=str, default='/home/cil/Documents/project/PPhorseMoshOpensource_submodule/dataset')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4, help='# of dataloadeer num_workers')
parser.add_argument('--num_epoch', type=int, default=100000, help='# of training epochs ')
parser.add_argument("--log_step", default=1000, type=int, help='log after n iters')
parser.add_argument("--val_step", default=1000, type=int, help='1000:log after n iters')
parser.add_argument("--save_step", default=2000, type=int, help='save models after n iters')

###### traning loss weights
parser.add_argument('--weight_loss_v2v', type=float, default=0.5)
parser.add_argument('--weight_loss_keypoints_3d', type=float, default=0.05)
parser.add_argument('--weight_loss_body_pose', type=float, default=0.001)
parser.add_argument('--weight_loss_pose_6d_ortho', type=float, default=0.1)
parser.add_argument('--weight_loss_global_orient', type=float, default=0.001)

#### diffusion model args
parser.add_argument("--num_diffusion_timesteps", default=50, type=int, help='total steps for diffusion')
parser.add_argument('--timestep_respacing_eval', type=str, default='ddim5', choices=['ddim5', 'ddpm'], help='ddim/ddpm sampling schedule')
parser.add_argument("--eval_spacing", default=20, type=int, help='downsample val set by # for faster evaluation during training')
parser.add_argument('--cond_mask_prob', type=float, default=0.01, help='by what prob to mask out conditions during training')
parser.add_argument('--shuffle', default='True', type=lambda x: x.lower() in ['true', '1'], help='shuffle in train dataloader')

#### dataset args
parser.add_argument("--ID", default=1, type=int, )
parser.add_argument('--dataset_file_train', type=str, default='20201128_ID_1_0007')
parser.add_argument('--dataset_file_val', type=str, default='20201128_ID_1_0010')

parser.add_argument('--do_augment', action="store_true", help="data agument or not")
args = parser.parse_args()


torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())

def collate_fn(item):
    try:
        item = default_collate(item)
    except Exception as e:
        import pdb;
        pdb.set_trace()
    return item

def get_dataset(logdir):
    model_cfg = get_config(args.model_cfg)
    # Create dataset and data loader
    # logdir = '/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/dataloaders/data_process'
    train_dataset = DatasetPFERD(cfg=model_cfg,
                                 dataset_file=args.dataset_file_train,
                                 data_root=args.dataset_root, ID = args.ID,
                                 train=True, split='train',spacing = 4,
                                 device=device,
                                 do_augment=args.do_augment,
                                 get_diffuse_feature=True, body_rep_stats_dir=logdir)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn)
    train_dataloader_iter = iter(train_dataloader)

    val_dataset = DatasetPFERD(cfg=model_cfg,
                                 dataset_file=args.dataset_file_val,
                                 data_root=args.dataset_root, ID = args.ID,
                                 train=False, split='val',spacing = 4,
                                 device=device,
                                 do_augment=False,
                                 get_diffuse_feature=True, body_rep_stats_dir=logdir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, train_dataloader_iter, train_dataset, val_dataloader, val_dataset

def train(writer, logger, logdir):
    model_cfg = get_config(args.model_cfg)

    train_dataloader, train_dataloader_iter, train_dataset, val_dataloader, val_dataset = get_dataset(logdir)
    # Setup model
    preprocess_stats = np.load(os.path.join(logdir, f'preprocess_stats/{args.ID}_{args.dataset_file_train}_preprocess_stats.npz'))
    body_rep_mean = torch.from_numpy(preprocess_stats['Xmean']).float().to(device)
    body_rep_std = torch.from_numpy(preprocess_stats['Xstd']).float().to(device)

    model = hSMALDiffusion(cfg=model_cfg, device=device, body_rep_mean=body_rep_mean, body_rep_std=body_rep_std,
                   weight_loss_v2v=args.weight_loss_v2v,weight_loss_body_pose=args.weight_loss_body_pose,
                   weight_loss_pose_6d_ortho=args.weight_loss_pose_6d_ortho,weight_loss_global_orient = args.weight_loss_global_orient,
                   cond_mask_prob=args.cond_mask_prob)
    diffusion_train = create_gaussian_diffusion(num_diffusion_timesteps=args.num_diffusion_timesteps, timestep_respacing='',
                                                body_rep_mean=body_rep_mean, body_rep_std=body_rep_std)
    schedule_sampler = UniformSampler(diffusion_train)
    diffusion_eval = create_gaussian_diffusion(num_diffusion_timesteps=args.num_diffusion_timesteps,
                                               timestep_respacing=args.timestep_respacing_eval,
                                               body_rep_mean=body_rep_mean, body_rep_std=body_rep_std)

    smal_model = SMAL(
        '/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl',
        device=torch.device('cpu'), used_betas=9)
    faces_cpu = smal_model.faces.cpu().data.numpy()
    render = COLORRenderer(faces= faces_cpu)

    # optimizer
    model.init_optimizers()
    total_steps = 0
    best_loss_keypoints_3d, best_loss = 10000, 10000
    for epoch in range(args.num_epoch):
        for step in tqdm(range(len(train_dataset) // args.batch_size)):
            total_steps += 1
            ### iter over train loader and mocap data loader
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)

            for param_name in batch.keys():
                if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                    batch[param_name] = batch[param_name].to(device)
            for param_name in batch['smpl_params'].keys():
                batch['smpl_params'][param_name] = batch['smpl_params'][param_name].to(device)

            batch_size = batch['smpl_params']['betas'].shape[0]
            t, weights = schedule_sampler.sample(batch_size, device)
            output = diffusion_train.training_losses(model, batch, t, epoch)
            ####################### log train loss ############################
            if total_steps % args.log_step == 0:
                for key in output['losses'].keys():
                    writer.add_scalar('train/{}'.format(key), output['losses'][key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [train]  {}: {:.10f}'. \
                        format(step, epoch, key, output['losses'][key].item())
                    logger.info(print_str)
                    # print(print_str)
            if total_steps % args.log_step == 0:
                verts = output['pred_vertices']
                color_rgb = render.render_front_view(verts=verts.cpu().data.numpy()[[0], ...])
                writer.add_image('train/image', torch.from_numpy(color_rgb/255.).permute(2,0,1), total_steps)

            ####################### log val loss #################################
            if total_steps % args.val_step == 0:
                val_loss_dict = {}
                # joint_vis_num = 0
                total_sample_num = 0
                with torch.no_grad():
                    for test_step, test_batch in tqdm(enumerate(val_dataloader)):
                        for param_name in test_batch.keys():
                            if param_name not in ['imgname', 'smpl_params', 'has_smpl_params',  'smpl_params_is_axis_angle']:
                                test_batch[param_name] = test_batch[param_name].to(device)
                        for param_name in test_batch['smpl_params'].keys():
                            test_batch['smpl_params'][param_name] = test_batch['smpl_params'][param_name].to(device)

                        # val_output = model.validation_step(test_batch, epoch)
                        shape = list(batch['x_t'].shape)
                        shape[0] = test_batch['smpl_params']['body_pose'].shape[0]
                        val_output = diffusion_eval.val_losses(model=model, batch=test_batch, shape=shape,
                                                               progress=False, clip_denoised=False,
                                                               cur_epoch=epoch,
                                                               timestep_respacing=args.timestep_respacing_eval)

                        # joint_vis_num += val_output['joint_vis_num_batch'].item()
                        total_sample_num += test_batch['smpl_params']['body_pose'].shape[0]
                        for key in val_output['losses'].keys():
                            if test_step == 0:
                                val_loss_dict[key] = val_output['losses'][key].detach().clone()
                            else:
                                val_loss_dict[key] += val_output['losses'][key].detach().clone()

                        if test_step == 0:
                            val_verts = val_output['pred_vertices']
                            color_rgb = render.render_front_view(verts=val_verts.cpu().data.numpy()[[0], ...])
                            writer.add_image('val/image', torch.from_numpy(color_rgb / 255.).permute(2, 0, 1), total_steps)

                for key in val_loss_dict.keys():
                    val_loss_dict[key] = val_loss_dict[key] / (test_step + 1)
                    writer.add_scalar('val/{}'.format(key), val_loss_dict[key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'. \
                        format(step, epoch, key, val_loss_dict[key].item())
                    logger.info(print_str)
                    print(print_str)

                if val_loss_dict['loss'] < best_loss:
                    best_loss = val_loss_dict['loss']
                    save_path = os.path.join(writer.file_writer.get_logdir(), "best_model_vis.pt")
                    state = {
                        "state_dict": model.state_dict(),
                    }
                    torch.save(state, save_path)
                    logger.info('[*] best model-vis saved\n')
                    print('[*] best model-vis saved\n')

            ################### save trained model #######################
            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "last_model.pt")
                state = {
                    "state_dict": model.state_dict(),
                }
                torch.save(state, save_path)
                logger.info('[*] last model saved\n')
                print('[*] last model saved\n')

if __name__ == '__main__':
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()
    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    shutil.copyfile(args.model_cfg, os.path.join(logdir, args.model_cfg.split('/')[-1]))
    train(writer, logger, logdir)
