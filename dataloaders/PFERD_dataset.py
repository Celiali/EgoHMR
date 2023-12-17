# import matplotlib
# matplotlib.use('TkAgg')
# Top of main python script
from typing import Dict
from yacs.config import CfgNode
from os.path import basename
import pickle as pkl
import smplx, glob
import pandas as pd
from torch.utils import data

from dataloaders.augmentation import hSMAL_dataaugment
from utils.other_utils import *
from utils.geometry import *


class DatasetPFERD(data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 data_root: str,
                 ID=None,
                 train: bool = True,
                 split='train',
                 spacing=1,
                 device=None,
                 do_augment=False,
                 get_diffuse_feature=False,
                 body_rep_stats_dir='',
                 ):
        """
        Dataset class used for loading images and corresponding annotations.
        """
        super(DatasetPFERD, self).__init__()
        self.train = train
        self.split = split
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment


        self.data_root = data_root
        self.ID = ID
        self.dataset_file = dataset_file
        self.spacing = spacing

        self.read_data(data_root, ID, dataset_file)

        ######## get mean/var for body representation feature in EgoHMR(to normalize for diffusion model)
        if get_diffuse_feature and split == 'train' and self.train:
            # 144-d
            global_orient = torch.from_numpy(self.global_orient).float()
            body_pose_all = torch.from_numpy(self.body_pose).float()
            full_pose_aa_all = torch.cat([global_orient, body_pose_all], dim=1).reshape(-1, 36, 3)  # [n, 24, 3]
            full_pose_rotmat_all = aa_to_rotmat(full_pose_aa_all.reshape(-1, 3)).view(-1, 36, 3, 3)  # [bs, 24, 3, 3]
            full_pose_rot6d_all = rotmat_to_rot6d(full_pose_rotmat_all.reshape(-1, 3, 3),
                                                  rot6d_mode='diffusion').reshape(-1, 36, 6).reshape(-1, 36 * 6)  # [n, 144]
            full_pose_rot6d_all = full_pose_rot6d_all.detach().cpu().numpy()
            Xmean = full_pose_rot6d_all.mean(axis=0)  # [d]
            Xstd = full_pose_rot6d_all.std(axis=0)  # [d]
            stats_root = os.path.join(body_rep_stats_dir, 'preprocess_stats')
            os.makedirs(stats_root) if not os.path.exists(stats_root) else None
            Xstd[0:6] = Xstd[0:6].mean() / 1.0  # for global orientation
            Xstd[6:] = Xstd[6:].mean() / 1.0  # for body pose
            np.savez_compressed(os.path.join(stats_root, f'{ID}_{dataset_file}_preprocess_stats.npz'), Xmean=Xmean, Xstd=Xstd)
            print('[INFO] mean/std for body_rep saved.')

        #self.smpl = smplx.create('data/smpl', model_type='smpl', gender='male')
        #print('[INFO] find {} samples in {}.'.format(self.dataset_len, dataset_file))

    def __len__(self) -> int:
        return self.body_pose.shape[0]

    def read_data(self, data_root, ID = None, dataset_file = None):
        if dataset_file is None:
            filelist = sorted(glob.glob(os.path.join(data_root, f'ID_{ID}', 'MODEL_DATA', f'*_hsmal.npz')))
        elif ID is None:
            filelist = sorted(glob.glob(os.path.join(data_root, '**', f'*_hsmal.npz')))
        else:
            filelist = [os.path.join(data_root, f'ID_{ID}', 'MODEL_DATA', f'{dataset_file}_hsmal.npz')]

        # Initialize dictionaries to hold the accumulated arrays
        self.global_orient, self.transl, self.body_pose, self.betas = [],[],[],[]
        # Iterate over the file paths
        for file_path in filelist:
            with np.load(file_path) as npz_file:
                # For each key, append the array to the corresponding list
                self.betas.append(npz_file['betas'][None].repeat(npz_file['trans'].shape[0], axis=0)[::self.spacing,])
                self.transl.append(npz_file['trans'][::self.spacing, :])
                self.global_orient.append(npz_file['poses'][::self.spacing, :3])
                self.body_pose.append(npz_file['poses'][::self.spacing,3:])

        # Stack the arrays for each key
        self.global_orient = np.vstack(self.global_orient).astype(float)
        self.body_pose = np.vstack(self.body_pose).astype(float)
        transl = np.vstack(self.transl).astype(float)
        betas = np.vstack(self.betas).astype(float)
        self.transl = np.zeros_like(transl)
        self.betas = np.zeros_like(betas)[:,:9]


    def __getitem__(self, idx: int) -> Dict:
        global_orient = self.global_orient[idx]
        transl = self.transl[idx]
        body_pose = self.body_pose[idx]
        betas = self.betas[idx]

        # keypoints_3d = self.keypoints_3d_pv[idx][0:24].copy()  # [24, 3], smpl joints

        smpl_params = {'global_orient': global_orient,
                       'transl': transl,
                       'body_pose': body_pose,
                       'betas': betas
                      }
        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }

        #################################### data augmentation
        augm_config = self.cfg.DATASETS.CONFIG

        smpl_params, has_smpl_params = hSMAL_dataaugment(smpl_params = smpl_params, has_smpl_params = has_smpl_params, \
                                                         do_augment = self.do_augment, augm_config = augm_config)
        item = {}
        ###### 3d joints
        # item['keypoints_3d'] = keypoints_3d_crop_auge.astype(np.float32)  # [24, 3]
        # item['keypoints_3d_full'] = keypoints_3d_full_auge.astype(np.float32)

        ###### smpl params
        item['smpl_params'] = smpl_params
        for key in item['smpl_params'].keys():
            item['smpl_params'][key] = item['smpl_params'][key].astype(np.float32)
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle

        return item

if __name__ == '__main__':
    from configs import get_config
    from tqdm import tqdm
    import sys
    sys.path.append('/home/cil/Documents/project/code_from_others/Smplx/EgoHMR')
    from render.mv import MeshViewer
    from render.renderer import COLORRenderer
    from render.SMAL.smal_torch.smal_torch import SMAL
    model_cfg = get_config("/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/configs/hSMALdiffusion.yaml")
    device = torch.device('cuda:0')

    class Options:
        def __init__(self,):
            self.dataset_root = '/home/cil/Documents/project/PPhorseMoshOpensource_submodule/dataset'
            self.do_augment = False
            self.batch_size = 4

    args = Options()
    logdir = '/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/dataloaders/data_process'
    train_dataset = DatasetPFERD(cfg=model_cfg,
                                 dataset_file='20201128_ID_1_0008',
                                 data_root=args.dataset_root, ID = '1',
                                 train=True, split='train',spacing = 4,
                                 device=device,
                                 do_augment=args.do_augment,
                                 get_diffuse_feature=True, body_rep_stats_dir=logdir)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, pin_memory=False)
    train_dataloader_iter = iter(train_dataloader)

    # load model
    smal_model = SMAL('/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl', device=device,used_betas=9)
    faces_cpu = smal_model.faces.cpu().data.numpy()

    mv = MeshViewer(width=1200, height=800,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 registered_keys=None,
                 render_flags = True, add_ground_plane = False, add_origin = False, y_up = False)
    # render = COLORRenderer(faces= faces_cpu)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.ion()
    for step in tqdm(range(len(train_dataset)// args.batch_size)):
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
            print(batch['smpl_params'][param_name].shape)

        verts, joints, Rs = smal_model(beta=batch['smpl_params']['betas'],  #
                                       theta=torch.cat([batch['smpl_params']['global_orient'], batch['smpl_params']['body_pose']], dim=1),
                                       trans=torch.zeros_like(batch['smpl_params']['transl']).to(device)
                                       )

        mv.update_multi_mesh(vertices=verts.cpu().data.numpy(), faces=faces_cpu, points=None)
        # color_rgb =render.render_front_view(verts=verts.cpu().data.numpy()[[0],...])
        # plt.cla()
        # plt.imshow(color_rgb)
        # plt.show()
        # plt.pause(0.01)



