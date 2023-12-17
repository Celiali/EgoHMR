import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import smplx
from smplx.utils import SMPLOutput

from models.resnet import resnet
from models.respointnet import ResnetPointnet
from models.egohmr.modulated_gcn.modulated_gcn import ModulatedGCN
from models.egohmr.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from utils.geometry import aa_to_rotmat, perspective_projection, rot6d_to_rotmat
from utils.konia_transform import rotation_matrix_to_angle_axis
from utils.other_utils import hSMAL_EDGES
from models.smal import HSMAL


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class hSMALDiffusion(nn.Module):
    def __init__(self, cfg, device=None,
                 body_rep_mean=None, body_rep_std=None,
                 weight_loss_v2v=0, weight_loss_body_pose=0,  weight_loss_pose_6d_ortho=0,weight_loss_global_orient=0,
                 cond_mask_prob=0,
                 diffusion_blk=4, gcn_dropout=0.0, gcn_nonlocal_layer=False, gcn_hid_dim=1024,
                 diffuse_fuse=False,
                 ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.body_rep_mean = body_rep_mean
        self.body_rep_std = body_rep_std
        self.diffuse_feat_dim = 6  # 6d rotation

        self.cond_mask_prob = cond_mask_prob
        self.diffuse_fuse = diffuse_fuse

        self.input_process_out_dim = 512
        self.input_process = InputProcess(self.diffuse_feat_dim, self.input_process_out_dim).to(self.device)
        self.timestep_embed_dim = 512
        self.sequence_pos_encoder = PositionalEncoding(self.timestep_embed_dim, dropout=0.1).to(self.device)
        self.embed_timestep = TimestepEmbedder(self.timestep_embed_dim, self.sequence_pos_encoder).to(self.device)

        ##### denoiser model
        edges = np.array(hSMAL_EDGES, dtype=np.int32)
        data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
        adj_mx = sp.coo_matrix((data, (i, j)), shape=(36, 36), dtype=np.float32)
        # build symmetric adjacency matrix
        adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
        adj_mx = normalize(adj_mx)  # + sp.eye(adj_mx.shape[0]))
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)  # [24, 24]
        adj_mx = adj_mx * (1 - torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])
        adj_mx = adj_mx.to(self.device)
        self.diffusion_model = ModulatedGCN(adj=adj_mx,
                                            in_dim=self.input_process_out_dim + self.timestep_embed_dim +1,
                                            hid_dim=gcn_hid_dim, out_dim=self.diffuse_feat_dim,
                                            num_layers=diffusion_blk, p_dropout=gcn_dropout,
                                            nonlocal_layer=gcn_nonlocal_layer).to(self.device)

        ######### Instantiate SMPL model
        # self.smpl = smplx.create('data/hsmal', model_type='hsmal').to(self.device)
        self.smal = HSMAL(
            model_path='/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl',
            model_type='hsmal',
            num_betas=9).to(self.device)
        ##### Define loss functions
        self.v2v_loss = nn.L1Loss(reduction='none')
        self.smpl_parameter_loss = ParameterLoss()

        self.weight_loss_v2v = weight_loss_v2v
        self.weight_loss_body_pose = weight_loss_body_pose
        self.weight_loss_global_orient = weight_loss_global_orient
        self.weight_loss_pose_6d_ortho = weight_loss_pose_6d_ortho


    def init_optimizers(self):
        self.opt_params = list(self.diffusion_model.parameters()) + \
                          list(self.embed_timestep.parameters()) + list(self.input_process.parameters())
        self.optimizer = torch.optim.AdamW(params=self.opt_params,
                                           lr=self.cfg.TRAIN.LR,
                                           weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

    def mask_cond(self, cond, force_mask=False):
        bs, J, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond  [bs, 1]
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, batch, timesteps, eval_with_uncond=True):
        # timesteps: [batch_size] (int)
        batch_size = timesteps.shape[0] #TODO: just get the batch_size batch['smpl_params']['betas'].shape[0]

        ############## timestep encoding
        timestep_emb = self.embed_timestep(timesteps).squeeze(0)  # [bs, d]  # timesteps: [bs]
        timestep_emb = timestep_emb.unsqueeze(1).repeat(1, 36, 1)   # [bs, 24, 512]

        # Sample 36 values from a Gaussian distribution
        # possible_mask = torch.randn((batch_size, 36)) > 0.8 #[bs, 36]
        # possible_mask[:, 0] = True  # pelvis: set global R always as visible
        # batch['vis_mask_smpl'] = possible_mask
        # conditioning_feats =possible_mask.unsqueeze(-1)#.repeat(1, 36, 1)  # [bs, 36, 1]
        ############### final condition: mask condition with p=cond_mask_prob
        # conditioning_feats_masked = self.mask_cond(conditioning_feats,
        #                                            force_mask=False)

        conditioning_feats_masked = torch.ones((batch_size, 36, 1)).to(timestep_emb.device)
        ############# pass to denoising model
        output = {}
        x_t = batch['x_t']  # noisy full_pose_rot6d  [bs, 36*6]
        x_t = x_t.reshape(batch_size, 36, -1)  # [bs, 36, 6]
        x_t_feat = self.input_process(x_t)  # [bs, 36, latent_dim=512]

        diffuse_feat = torch.cat([conditioning_feats_masked, x_t_feat, timestep_emb], axis=-1)  # [bs, 24, (2048+512+3+6)+512+512]
        diffuse_output = self.diffusion_model(diffuse_feat)  # body pose 6d [bs, 24, 6]

        # if self.diffuse_fuse:
        #     if eval_with_uncond:
        #         # generation with only scene conditions, without image condition
        #         conditioning_feats_masked_all = self.mask_cond(conditioning_feats,
        #                                                        force_mask=True)  # mask conditions for all data
        #         diffuse_feat_uncond = torch.cat([conditioning_feats_masked_all, x_t_feat, timestep_emb], axis=-1)
        #         diffuse_output_uncond = self.diffusion_model(diffuse_feat_uncond)
        #         diffuse_output_cond = diffuse_output.clone()
        #         guidance_param = 0
        #         diffuse_output = diffuse_output_uncond + guidance_param * (diffuse_output - diffuse_output_uncond)
        #         # replace joint rotation for visible joints by generated results also conditioned on image feature
        #         vis_mask_body_pose_6d = conditioning_feats.unsqueeze(-1).repeat(1, 1, 6).reshape(batch_size, -1)  # [bs, 144]
        #         diffuse_output = diffuse_output.reshape(batch_size, -1)
        #         diffuse_output_cond = diffuse_output_cond.reshape(batch_size, -1)
        #         diffuse_output[vis_mask_body_pose_6d] = diffuse_output_cond[vis_mask_body_pose_6d]

        diffuse_output = diffuse_output.reshape(batch_size, -1)  # [bs, 36 6]  -> [bs, 216]
        output['pred_x_start'] = diffuse_output
        diffuse_output = diffuse_output * self.body_rep_std + self.body_rep_mean
        pred_pose_6d = diffuse_output
        pred_pose_rotmat = rot6d_to_rotmat(pred_pose_6d, rot6d_mode='diffusion').view(batch_size, 36, 3, 3)  # [bs, 36, 3, 3]

        ############## Store useful regression outputs to the output dict
        pred_smpl_params = {'global_orient': pred_pose_rotmat[:, [0]],
                            'body_pose': pred_pose_rotmat[:, 1:],
                            }

        #  global_orient: [bs, 1, 3, 3], body_pose: [bs, 23, 3, 3], shape...
        output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
        output['pred_pose_6d'] = pred_pose_6d

        self.smal_output = self.smal(**{k: v.float() for k, v in pred_smpl_params.items()}, return_full_pose=True, pose2rot=False)
        output['pred_vertices'] = self.smal_output.vertices  # [bs, 6890, 3]
        output['pred_keypoints_3d'] = self.smal_output.joints  # [bs, 45, 3]
        return output



    def compute_loss(self, batch, output, cur_epoch=0):
        pred_smpl_params = output['pred_smpl_params']
        pred_pose_6d = output['pred_pose_6d']
        batch_size = pred_smpl_params['body_pose'].shape[0]
        pred_keypoints_3d = output['pred_keypoints_3d']  # [bs, 24, 3]
        # Get annotations
        gt_smpl_params = batch['smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        ####### compute v2v loss
        gt_smal_output = self.smal(**{k: v.float() for k, v in gt_smpl_params.items()}, return_full_pose=True, pose2rot=True)
        gt_vertices = gt_smal_output.vertices  # smplx vertices
        gt_joints = gt_smal_output.joints
        pred_vertices = output['pred_vertices']  # [bs, 6890, 3]
        loss_v2v = self.v2v_loss(pred_vertices - pred_keypoints_3d[:, [0], :].clone(),
                                 gt_vertices - gt_joints[:, [0], :].clone()).mean()

        ########### Compute loss on SMPL parameters (pose in rot mat form 3x3)
        # loss_smpl_params: keys: ['global_orient'(bs, n_sample, 1, 3, 3), 'body_pose'(bs, n_sample, 23, 3, 3), 'betas']
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            if k != 'transl':
                gt = gt_smpl_params[k]
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)  # [bs, 1/23, 3, 3]
                ## MSE loss for rotation/shape
                loss_smpl_params[k] = self.smpl_parameter_loss(pred, gt).sum()/batch_size

        ########### Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 3, 2)  # different 6d order from prohmr code
        loss_pose_6d_ortho = (torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=self.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2
        loss_pose_6d_ortho = loss_pose_6d_ortho.mean()

        loss = self.weight_loss_v2v * loss_v2v + \
               self.weight_loss_body_pose * loss_smpl_params['body_pose'] + \
               self.weight_loss_global_orient * loss_smpl_params['global_orient'] + \
               self.weight_loss_pose_6d_ortho * loss_pose_6d_ortho
        # self.weight_loss_betas * loss_smpl_params['betas'] + \

        losses = dict(loss=loss.detach(),
                      loss_v2v=loss_v2v.detach(),
                      loss_body_pose=loss_smpl_params['body_pose'].detach(),
                      loss_pose_6d_ortho=loss_pose_6d_ortho.detach(),
                      )

        output['losses'] = losses

        return loss



    def training_step(self, batch, timesteps, cur_epoch):
        self.training = True
        self.diffusion_model.train()

        self.input_process.train()
        self.embed_timestep.train()

        ### forward step
        output = self.forward(batch, timesteps, eval_with_uncond=False)
        ### compute loss
        loss = self.compute_loss(batch, output, cur_epoch=cur_epoch)
        ### backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output


    def validation_setup(self):
        self.training = False
        self.diffusion_model.eval()

        self.input_process.eval()
        self.embed_timestep.eval()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [5000, 1]: 0, 1, 2, ..., 4999
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # torch.arange(0, d_model, 2): [256]: 0, 2, 4, 6, 8, ..., 510  div_term: [256]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)   # [5000, 1, 512]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)   # timesteps: [bs]


class InputProcess(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_dim, self.latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)
        return x


class FCHeadBeta(nn.Module):
    def __init__(self, in_dim=None, condition_on_pose=False, pose_dim=144):
        super(FCHeadBeta, self).__init__()
        self.condition_on_pose = condition_on_pose
        if self.condition_on_pose:
            in_dim = in_dim + pose_dim
        self.layers = nn.Sequential(nn.Linear(in_dim, 1024),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(1024, 10))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load('data/smpl_mean_params.npz')
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None]  # [1, 10]
        self.register_buffer('init_betas', init_betas)

    def forward(self, feats, pred_pose):
        if self.condition_on_pose:
            feats = torch.cat([feats, pred_pose], dim=-1)  # [bs, feat_dim+144]
        offset = self.layers(feats)  # [bs, 10]

        pred_betas = offset + self.init_betas
        return pred_betas


class TranslEnc(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super(TranslEnc, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim,64),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(64, out_dim))

    def forward(self, input):
        transl_feat = self.layers(input)
        return transl_feat