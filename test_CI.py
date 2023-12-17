import argparse
from tqdm import tqdm
from configs import get_config, prohmr_config
import smplx
import pandas as pd
import pickle as pkl
import random
# import randomnvidia
# from utils.pytorch3d_chamfer_distance import chamfer_distance


from utils.pose_utils import *
# from utils.renderer import *
from utils.other_utils import *
# from utils.geometry import *
# from dataloaders.egobody_dataset import DatasetEgobody
from dataloaders.PFERD_dataset import DatasetPFERD
# from utils.geometry import perspective_projection

# from models.egohmr.egohmr import EgoHMR
from models.egohmr.hSMALhmr import hSMALDiffusion
from diffusion.model_util import create_gaussian_diffusion
from models.smal import HSMAL
# from render.SMAL.smal_torch.smal_torch import SMAL


parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--dataset_root', type=str, default='/home/cil/Documents/project/PPhorseMoshOpensource_submodule/dataset', help='path to egobody dataset')
parser.add_argument('--checkpoint', type=str, default='/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/results/test/85679/last_model.pt', help='path to trained checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (configs/prohmr.yaml)')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for inference')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=100, help='How often to print results')
parser.add_argument("--seed", default=0, type=int)

####### save/render/vis args
parser.add_argument('--render', default='True', type=lambda x: x.lower() in ['true', '1'], help='render pred body mesh on images')
parser.add_argument('--render_multi_sample', default='True', type=lambda x: x.lower() in ['true', '1'], help='render all pred samples for input image')
parser.add_argument('--render_step', type=int, default=1, help='how often to render results')

parser.add_argument('--vis_o3d', default='False', type=lambda x: x.lower() in ['true', '1'], help='visualize 3d body and scene with open3d')
parser.add_argument('--vis_o3d_gt', default='False', type=lambda x: x.lower() in ['true', '1'], help='if visualize ground truth body as well')
parser.add_argument('--vis_step', type=int, default=8, help='how often to visualize 3d results')  # 8/1

#### diffusion model args
parser.add_argument("--num_diffusion_timesteps", default=50, type=int, help='total steps for diffusion')
parser.add_argument('--timestep_respacing_eval', type=str, default='ddpm', choices=['ddim5', 'ddpm'], help='ddim/ddpm sampling schedule')
parser.add_argument('--diffuse_fuse', default='True', type=lambda x: x.lower() in ['true', '1'], help='if to use classifier-free sampling')
parser.add_argument('--only_mask_img_cond', default='True', type=lambda x: x.lower() in ['true', '1'],
                    help='only mask img features during trainig with cond_mask_prob')
parser.add_argument('--pelvis_vis_loosen', default='True', type=lambda x: x.lower() in ['true', '1'],
                    help='set pelvis joint visibility the same as knees, allows more flexibility for lower body diversity')

#### eval args
parser.add_argument("--eval_spacing", default=1, type=int, help='downsample test set by #')
parser.add_argument('--num_samples', type=int, default=6, help='Number of test samples to draw')
parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'], help='shuffle in dataloader')

#### dataset args
parser.add_argument("--ID", default=1, type=int, )
parser.add_argument('--dataset_file_train', type=str, default='20201128_ID_1_0007')
parser.add_argument('--dataset_file_val', type=str, default='20201128_ID_1_0010')

args = parser.parse_args()

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
fixseed(args.seed)

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test():
    ############################## Load model config
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)
    # Update number of test samples drawn to the desired value
    model_cfg.defrost()
    model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
    model_cfg.freeze()


    ############################### Setup and load pretrained model, diffusion
    logdir = '/'.join(args.checkpoint.split('/')[0:-1])
    # Setup model
    preprocess_stats = np.load(
        os.path.join(logdir, f'preprocess_stats/{args.ID}_{args.dataset_file_train}_preprocess_stats.npz'))
    body_rep_mean = torch.from_numpy(preprocess_stats['Xmean']).float().to(device)
    body_rep_std = torch.from_numpy(preprocess_stats['Xstd']).float().to(device)

    model = hSMALDiffusion(cfg=model_cfg, device=device, body_rep_mean=body_rep_mean, body_rep_std=body_rep_std,
                           cond_mask_prob=0.0)

    if args.timestep_respacing_eval == 'ddpm':
        args.timestep_respacing_eval = ''
    diffusion_sample = create_gaussian_diffusion(num_diffusion_timesteps=args.num_diffusion_timesteps, timestep_respacing=args.timestep_respacing_eval,
                                          body_rep_mean=body_rep_mean, body_rep_std=body_rep_std)

    weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['state_dict'], strict=False)
    model.eval()
    print('load traind mode from:', args.checkpoint)
    print('diffusion sample method:', args.timestep_respacing_eval)

    ################################# load smpl models
    smal = HSMAL(
        model_path='/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl',
        model_type='hsmal',
        num_betas=9).to(device)

    #################################### create meshviewer
    from render.mv import MeshViewer
    mv = MeshViewer(width=1200, height=800,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 registered_keys=None,
                 render_flags = True, add_ground_plane = False, add_origin = False, y_up = True)
    xx, yy, zz = np.meshgrid(np.linspace(0, 2, args.batch_size), np.linspace(0, 2, args.num_samples),[0])
    grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    colors = (grid*255).astype(int)
    xx, yy, zz = np.meshgrid(np.linspace(-2, 2, args.batch_size), np.linspace(-2, 2,  args.num_samples),[0])
    grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    trans =grid*10.

    for step in tqdm(range(100)):
        ################################# test
        curr_batch_size = args.batch_size #batch['smpl_params']['betas'].shape[0]
        with torch.no_grad():
            ######## iterate for multiple samples
            shape = [curr_batch_size, 36*6]
            batch = {}
            out_all_samples = {}
            out_all_samples['pred_smpl_params'] = {}
            for n in range(args.num_samples):
                out_cur_sample = diffusion_sample.val_losses(model=model, batch=batch, shape=shape, progress=False,
                                                             clip_denoised=False, cur_epoch=0,
                                                             timestep_respacing=args.timestep_respacing_eval,cond_grad_weight=0.,
                                                             cond_fn_with_grad=False, compute_loss = False)
                for key in out_cur_sample['pred_smpl_params'].keys():
                    if key not in out_all_samples['pred_smpl_params'].keys():
                        out_all_samples['pred_smpl_params'][key] = []
                    out_all_samples['pred_smpl_params'][key].append(out_cur_sample['pred_smpl_params'][key].unsqueeze(1))
            for key in out_cur_sample['pred_smpl_params'].keys():
                out_all_samples['pred_smpl_params'][key] = torch.cat(out_all_samples['pred_smpl_params'][key], dim=1)  # [bs, n_sample, ...]

        ###### get pred smpl params
        pred_body_pose = out_all_samples['pred_smpl_params']['body_pose']  # [bs, n_sample, 35, 3, 3]
        pred_global_orient = out_all_samples['pred_smpl_params']['global_orient']  # [bs, n_sample, 1, 3, 3]
        if True:
            pred_global_orient = torch.from_numpy(np.expand_dims(np.eye(3), axis=[0,1,2]).repeat(pred_global_orient.shape[0], axis=0).repeat(pred_global_orient.shape[1], axis=1)).float().to(device)# [bs, n_sample, 1, 3, 3]

        ###### get pred smpl joints / vertices
        pred_output = smal(body_pose=pred_body_pose.reshape(-1, 35, 3, 3),
                                   global_orient=pred_global_orient.reshape(-1, 1, 3, 3), pose2rot=False)
        pred_vertices = pred_output.vertices.reshape(curr_batch_size, -1, 1497, 3)  # [bs, n_sample, 1497, 3]
        pred_keypoints_3d = pred_output.joints.reshape(curr_batch_size, -1, 36, 3)#[:, :, 0:24, :]  # [bs, n_sample, 24, 3]
        pred_pelvis = pred_keypoints_3d[:, :, [0], :].clone()  # [bs, n_sample, 1, 3]
        pred_keypoints_3d_align = pred_keypoints_3d - pred_pelvis
        pred_vertices_align = pred_vertices - pred_pelvis

        ##### get pred cam joints / vertices in full img coord
        pred_vertices_full = pred_vertices #+ batch['smpl_params']['transl'].unsqueeze(1).unsqueeze(1)  # [bs, n_sample, 6890, 3]
        pred_keypoints_3d_full = pred_keypoints_3d #+ batch['smpl_params']['transl'].unsqueeze(1).unsqueeze(1)  # [bs, n_sample, 24, 3]

        ########################## visualize
        ##### vis_multi_sample=True: render for all samples, otherwise only render one sample
        if args.render and step % args.render_step == 0:
            pred_vertices_full_numpy = pred_vertices_full.cpu().data.numpy()#[bs, nsample, 1497,3 ]
            translated_verts = pred_vertices_full_numpy.reshape(-1, 1497, 3) + np.expand_dims(trans, axis=1)
            mv.update_multi_mesh(vertices=translated_verts, faces=smal.faces, points=None, color=colors)


    '''
    ################################ Create dataset and data loader
    test_dataset = DatasetPFERD(cfg=model_cfg,
                               dataset_file=args.dataset_file_train,
                               data_root=args.dataset_root, ID=args.ID,
                               train=False, split='test', spacing=4,
                               device=device,
                               do_augment=False,
                               get_diffuse_feature=True, body_rep_stats_dir=logdir)
    dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    ################################# create list for evaluation metrics
    # accuracy
    g_mpjpe_all = np.zeros([len(test_dataset), args.num_samples])
    mpjpe_all = np.zeros([len(test_dataset), args.num_samples])
    pa_mpjpe_all = np.zeros([len(test_dataset), args.num_samples])
    v2v_all = np.zeros([len(test_dataset), args.num_samples])  # translation/pelv-aligned
    # diversity
    std_joints_all = np.zeros([len(test_dataset)])
    apd_joints_all = np.zeros([len(test_dataset)])

    pred_body_pose_list = []
    pred_global_orient_list = []
    ################################# test
    for step, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        curr_batch_size = args.batch_size #batch['smpl_params']['betas'].shape[0]
        with torch.no_grad():
            ######## iterate for multiple samples
            shape = [curr_batch_size, 36*6]
            out_all_samples = {}
            out_all_samples['pred_smpl_params'] = {}
            for n in range(args.num_samples):
                out_cur_sample = diffusion_sample.val_losses(model=model, batch=batch, shape=shape, progress=False,
                                                             clip_denoised=False, cur_epoch=0,
                                                             timestep_respacing=args.timestep_respacing_eval,cond_grad_weight=0.,
                                                             cond_fn_with_grad=False)
                for key in out_cur_sample['pred_smpl_params'].keys():
                    if key not in out_all_samples['pred_smpl_params'].keys():
                        out_all_samples['pred_smpl_params'][key] = []
                    out_all_samples['pred_smpl_params'][key].append(out_cur_sample['pred_smpl_params'][key].unsqueeze(1))
            for key in out_cur_sample['pred_smpl_params'].keys():
                out_all_samples['pred_smpl_params'][key] = torch.cat(out_all_samples['pred_smpl_params'][key], dim=1)  # [bs, n_sample, ...]

        ###### get gt annotations
        gt_pose = {}
        gt_pose['global_orient'] = batch['smpl_params']['global_orient'].to(device)
        gt_pose['transl'] = batch['smpl_params']['transl'].to(device)
        gt_pose['body_pose'] = batch['smpl_params']['body_pose'].to(device)
        gt_pose['betas'] = batch['smpl_params']['betas'].to(device)

        ###### get pred smpl params
        pred_body_pose = out_all_samples['pred_smpl_params']['body_pose']  # [bs, n_sample, 35, 3, 3]
        pred_global_orient = out_all_samples['pred_smpl_params']['global_orient']  # [bs, n_sample, 1, 3, 3]

        pred_body_pose_list.append(pred_body_pose)
        pred_global_orient_list.append(pred_global_orient)

        ###### get pred smpl joints / vertices
        pred_output = smal(body_pose=pred_body_pose.reshape(-1, 35, 3, 3),
                                   global_orient=pred_global_orient.reshape(-1, 1, 3, 3), pose2rot=False)
        pred_vertices = pred_output.vertices.reshape(curr_batch_size, -1, 1497, 3)  # [bs, n_sample, 1497, 3]
        pred_keypoints_3d = pred_output.joints.reshape(curr_batch_size, -1, 36, 3)#[:, :, 0:24, :]  # [bs, n_sample, 24, 3]
        pred_pelvis = pred_keypoints_3d[:, :, [0], :].clone()  # [bs, n_sample, 1, 3]
        pred_keypoints_3d_align = pred_keypoints_3d - pred_pelvis
        pred_vertices_align = pred_vertices - pred_pelvis

        ##### get pred cam joints / vertices in full img coord
        pred_vertices_full = pred_vertices #+ batch['smpl_params']['transl'].unsqueeze(1).unsqueeze(1)  # [bs, n_sample, 6890, 3]
        pred_keypoints_3d_full = pred_keypoints_3d #+ batch['smpl_params']['transl'].unsqueeze(1).unsqueeze(1)  # [bs, n_sample, 24, 3]

        ##### get gt body joints / vertices
        gt_body = smal(**gt_pose)
        gt_joints = gt_body.joints
        gt_vertices = gt_body.vertices
        gt_keypoints_3d = gt_joints[:, :36, :]  # [bs, 36, 3]
        gt_pelvis = gt_keypoints_3d[:, [0], :].clone()  # [bs,1,3]
        gt_keypoints_3d_align = gt_keypoints_3d - gt_pelvis
        gt_vertices_align = gt_vertices - gt_pelvis

        # # G-MPJPE
        # gmpjpe_per_joint = torch.sqrt(((pred_keypoints_3d_full - gt_keypoints_3d.unsqueeze(1)) ** 2).sum(dim=-1))  # [bs, n_sample, 24]
        # gmpjpe_cur_batch = gmpjpe_per_joint.mean(dim=-1).cpu().numpy()  # [bs, n_sample]
        # g_mpjpe_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = gmpjpe_cur_batch
        #
        # # MPJPE
        # mpjpe_per_joint = torch.sqrt(((pred_keypoints_3d_align - gt_keypoints_3d_align.unsqueeze(1)) ** 2).sum(dim=-1))  # [bs, n_sample, 24]
        # mpjpe_cur_batch = mpjpe_per_joint.mean(dim=-1).cpu().numpy()  # [bs, n_sample]
        # mpjpe_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = mpjpe_cur_batch
        #
        # # PA-MPJPE
        # n_joints = gt_keypoints_3d_align.shape[-2]
        # pampjpe_per_joint = reconstruction_error(
        #     pred_keypoints_3d_align.reshape(-1, n_joints, 3).cpu().numpy(),
        #     gt_keypoints_3d_align.unsqueeze(1).repeat(1, args.num_samples, 1, 1).reshape(-1, n_joints, 3).cpu().numpy(),
        #     avg_joint=False).reshape(curr_batch_size, args.num_samples, -1)  # [bs, num_samples, 24]
        # pampjpe_cur_batch = pampjpe_per_joint.mean(axis=-1)  # [bs, num_samples]
        # pa_mpjpe_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = pampjpe_cur_batch
        # # V2V
        # v2v_per_verts = torch.sqrt(((pred_vertices_align - gt_vertices_align.unsqueeze(1)) ** 2).sum(dim=-1))  # [bs, n_sample, 6890]
        # v2v_cur_batch = v2v_per_verts.mean(dim=-1).cpu().numpy()
        # v2v_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = v2v_cur_batch
        # ############ diversity std joints
        # # pred_keypoints_3d_align  # [bs, n_samples, 36, 3]
        # std_joints_cur_batch = torch.std(pred_keypoints_3d_align, dim=1, unbiased=True).mean(dim=-1).mean(dim=-1).cpu().numpy()  # [bs]
        # std_joints_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = std_joints_cur_batch
        #
        # ############### diversity apd joints
        # # pred_keypoints_3d_align  # [bs, n_samples, 24, 3]
        # a = pred_keypoints_3d_align.detach().cpu().numpy()
        # n_samples = a.shape[1]
        # pairwise_dist = np.linalg.norm(a[:, None, :, :, :] - a[:, :, None, :, :], axis=-1)  # [bs, n_samples, n_samples, 24]
        # apd_joints = pairwise_dist.sum(axis=(-1, -2, -3)) / a.shape[-2] / n_samples / (n_samples - 1) / 2  # [bs]
        # apd_joints_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = apd_joints
        #
        # if step % args.log_freq == 0 and step > 0:
        #     ######## compute from N random samples
        #     print('--------- mode: compute mean from {} random samples ---------'.format(args.num_samples))
        #     error_dict = {
        #         'G-MPJPE': 1000 * g_mpjpe_all[:step * args.batch_size].mean(),
        #         'MPJPE': 1000 * mpjpe_all[:step * args.batch_size].mean(),
        #         'PA-MPJPE': 1000 * pa_mpjpe_all[:step * args.batch_size, ].mean(),
        #         'V2V': 1000 * v2v_all[:step * args.batch_size].mean(),
        #         }
        #
        #     print('G-MPJPE all: {:0.2f}'.format(error_dict['G-MPJPE']))
        #     print('MPJPE all: {:0.2f}'.format(error_dict['MPJPE']))
        #     print('PA-MPJPE all: {:0.2f}'.format(error_dict['PA-MPJPE']))
        #     print('V2V all: {:0.2f}'.format(error_dict['V2V']))
        #
        #     ####### mean std over all data/all samples/all joints/xyz coords
        #     print('--------- diversity ---------')
        #     diversity_dict = {
        #         'std-joints': 1000 * std_joints_all[:step * args.batch_size].mean(),
        #         'apd-joints': 1000 * apd_joints_all[:step * args.batch_size].mean(),
        #     }
        #     print('std-joints all: {:0.2f}'.format(diversity_dict['std-joints']))
        #     print('apd-joints all: {:0.2f}'.format(diversity_dict['apd-joints']))

        ########################## visualize
        ##### vis_multi_sample=True: render for all samples, otherwise only render one sample
        if args.render and step % args.render_step == 0:
            if args.render_multi_sample:
                vis_sample_n = args.num_samples
            else:
                vis_sample_n = 1

            pred_vertices_full_numpy = pred_vertices_full.cpu().data.numpy()#[bs, nsample, 1497,3 ]
            # # Calculate the extent of each point cloud
            # max_extent = np.max(pred_vertices_full_numpy, axis=(2, 3)) - np.min(pred_vertices_full_numpy, axis=(2, 3))
            # # Determine the spacing between point clouds
            # spacing = np.max(max_extent) * 1.2  # 1.2 for 20% extra space
            # # Calculate translations
            # translations = np.zeros_like(pred_vertices_full_numpy)
            # for i in range(pred_vertices_full_numpy.shape[0]):
            #     for j in range(vis_sample_n):
            #         translations[i, j, :, 0] = (-i-int(pred_vertices_full_numpy.shape[0]/2)) * spacing  # Translate along Y-axis (if needed)
            #         translations[i, j, :, 1] = (j - int(vis_sample_n/2)) * spacing  # Translate along X-axis
            # Apply translations
            # translated_verts = pred_vertices_full_numpy + translations
            translated_verts = pred_vertices_full_numpy.reshape(-1, 1497, 3) + np.expand_dims(trans, axis=1)
            mv.update_multi_mesh(vertices=translated_verts, faces=smal.faces, points=None, color=colors)
    '''

    # print('*** Final Results ***')
    # print('--------- mode: compute mean from {} random samples ---------'.format(args.num_samples))
    # error_dict = {
    #     'G-MPJPE': 1000 * g_mpjpe_all.mean(),
    #     'MPJPE': 1000 * mpjpe_all.mean(),
    #     'PA-MPJPE': 1000 * pa_mpjpe_all.mean(),
    #     'V2V': 1000 * v2v_all.mean(),
    #     }
    # print('G-MPJPE all/vis/invis: {:0.2f}'.format(error_dict['G-MPJPE']))
    # print('MPJPE all/vis/invis: {:0.2f}'.format(error_dict['MPJPE']))
    # print('PA-MPJPE all/vis/invis: {:0.2f}'.format(error_dict['PA-MPJPE']))
    # print('V2V all/vis/invis: {:0.2f}'.format(error_dict['V2V']))
    #
    # ####### mean std over all data/all samples/all joints/xyz coords
    # print('--------- diversity ---------')
    # diversity_dict = {
    #     'std-joints': 1000 * std_joints_all.mean(),
    #      'apd-joints': 1000 * apd_joints_all.mean(),
    #     }
    # print('std-joints all/vis/invis: {:0.2f}'.format(diversity_dict['std-joints']))
    # print('apd-joints all/vis/invis: {:0.2f}'.format(diversity_dict['apd-joints']))


if __name__ == '__main__':
    test()


