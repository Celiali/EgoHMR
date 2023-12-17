"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smal_torch')
import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl 
from batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from smal_basics import align_smal_template_to_symmetry_axis#,get_horse_template

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(object):
    def __init__(self, model_path, device, use_smal_betas=False,used_betas=None, dtype=torch.float): #opts,
        # self.opts = opts
        self.device = device #torch.device("cuda:{}".format(str(opts.gpu)) if torch.cuda.is_available() and opts.gpu != "cpu" else "cpu")
        self.used_betas = used_betas
        # -- Load SMPL params --
        with open(model_path, 'rb') as f:
            dd = pkl.load(f, encoding="latin1")
            # print(dd.keys())

        self.faces = torch.from_numpy(dd['f'].astype(np.int32)).type(torch.int32).to(self.device)

        v_template = dd['v_template']
        v, self.left_inds, self.right_inds, self.center_inds, symIdx = align_smal_template_to_symmetry_axis(model_path, v_template)

        # symIdx
        self.symIdx = Variable(
            torch.tensor(symIdx).long().to(self.device), #.cuda(device=self.opts.gpu_id), https://blog.csdn.net/CV_YOU/article/details/84592065
            requires_grad=False)

        # Mean template vertices
        self.v_template = Variable(
            torch.from_numpy(v).type(torch.float32).to(self.device), #.cuda(device=self.opts.gpu_id), https://blog.csdn.net/CV_YOU/article/details/84592065
            requires_grad=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis

        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        #https://github.com/pytorch/pytorch/issues/47160
        self.shapedirs = Variable(
            torch.from_numpy(shapedir.copy()).type(torch.float32).to(self.device),requires_grad=False)#.cuda(device=self.opts.gpu_id),

        # Regressor for joint locations given shape
        self.J_regressor = Variable(
            torch.from_numpy(dd['J_regressor'].T.todense()).type(torch.float32).to(self.device), #.cuda(device=self.opts.gpu_id),
            requires_grad=False)

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]

        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = Variable(
            torch.from_numpy(posedirs.copy()).type(torch.float32).to(self.device),requires_grad=False) #.cuda(device=self.opts.gpu_id),

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = Variable(
            torch.from_numpy(undo_chumpy(dd['weights'])).type(torch.float32).to(self.device), #.cuda(device=self.opts.gpu_id),
            requires_grad=False)

    def __call__(self, beta, theta, trans=None, del_v=None, betas_logscale=None, get_skin=True):

        if self.used_betas is not None: #self.opts.use_smal_betas
            nBetas = self.used_betas
        else:
            nBetas = 9

        # 1. Add shape blend shapes

        if nBetas > 0:
            if del_v is None:
                v_shaped = self.v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = self.v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = self.v_template.unsqueeze(0)
            else:
                v_shaped = self.v_template + del_v

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # N x 36 x 3 x 3
        if len(theta.shape) ==4:
            Rs = theta
        else:
            Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3]), device=self.device), [-1, 36, 3, 3])
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(self.device),  [-1, 315])#.cuda(device=self.opts.gpu_id),



        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        #4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, device=self.device, betas_logscale=betas_logscale)


        # 5. Do skinning:
        num_batch = theta.shape[0]

        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 36])


        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 36, 16])),
                [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device = self.device)], 2)#.cuda(device=self.opts.gpu_id)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device = self.device) #.cuda(device=self.opts.gpu_id)

        verts = verts + trans[:,None,:]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def test_gpu(opts):
    from time import time
    import os
    if opts.gpu != 'cpu' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # print(device)

    pose_size = 108
    beta_size = 9

    np.random.seed(9608)
    print(os.path.abspath(__file__))
    model = SMAL(model_path='../smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl', device=device, use_smal_betas=False)#opts = opts)
    for i in range(10):
        pose = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 0.4) \
            .type(torch.float32).to(device)
        # pose = torch.from_numpy(np.zeros((1, pose_size))).type(torch.float32).to(device)
        betas = torch.from_numpy((np.random.rand(32, beta_size) - 0.5) * 0.06) \
            .type(torch.float32).to(device)
        s = time()
        trans = torch.from_numpy(np.zeros((32, 3))).type(torch.float32).to(device)
        result, joints, Rs = model(betas, pose, trans)
        print(time() - s)

        print("ok")
        # outmesh_path = './hosmal_torch_{}.obj'
        # for j in range(10):#result.shape[0]):
        #     model.write_obj(result[j], outmesh_path.format(j))

def view_model(opts):
    if opts.gpu != 'cpu' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SMAL(model_path='../smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl', device=device,
                 use_smal_betas=False)  # opts = opts)

    points = model.v_template.cpu().data.numpy()
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='x')
    ax.set_xlabel('x', )
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def view_model_left_right(opts):
    if opts.gpu != 'cpu' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SMAL(model_path='../smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl', device=device,
                 use_smal_betas=False)  # opts = opts)

    points = model.v_template.cpu().data.numpy()
    index_part1 = np.where(points[:,1] < 0)[0] # right range(813, 1497)
    index_part2 = np.where(points[:, 1]>= 0)[0] # left range(0, 813)

    all_faces = model.faces.cpu().data.numpy()
    index_part1_faces = [] #range(1,2991,2)
    index_part2_faces = []
    for f_index, f in enumerate(all_faces):
        right = np.where(f >= 813)
        if len(right[0]) !=0:
            index_part1_faces.append(f_index)
        else:
            index_part2_faces.append(f_index)

    print(index_part1_faces)
    print(index_part2_faces)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[index_part1, 0], points[index_part1, 1], points[index_part1, 2], c='r', marker='x')
    ax.scatter(points[index_part2, 0], points[index_part2, 1], points[index_part2, 2], c='g', marker='x')
    ax.set_xlabel('x', )
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# def test_rotation(opts):
#     import os
#     import torch
#     import numpy as np
#
#     if opts.gpu != 'cpu' and torch.cuda.is_available():
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#
#     from renderer.pr_renderer import MeshViewer
#     mv = MeshViewer()
#     import torch
#     from dataset import SoundDS
#
#     trainpath = '/home/cil/Documents/project/AudioPro/testaudio/data/train5'
#     traindataset = SoundDS(path=trainpath)
#     train_dl = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=False)
#     print(len(train_dl))
#
#     model = SMAL(model_path='../smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl', device=device,
#                  use_smal_betas=False)
#     betas = torch.from_numpy(np.zeros((1, 9))).type(torch.float32).to(device)
#     trans = torch.from_numpy(np.zeros((1, 3))).type(torch.float32).to(device)
#     poses_input = torch.from_numpy(np.zeros((1, 108))).type(torch.float32).to(device)
#     for data in train_dl:
#         index, spec, pose = data
#
#         poses_input[0,:84] = (pose[0,0,:]).unsqueeze(0)/100.
#         result, joints, Rs = model(betas, poses_input, trans)
#         verts = result.squeeze().detach().cpu().numpy()
#         faces = model.faces.detach().cpu().numpy()
#
#         pose2 = axis_angle_to_matrix(torch.reshape(poses_input,(-1,3)))
#         rot6d = matrix_to_rotation_6d(pose2)
#         rotmat_recon = rotation_6d_to_matrix(rot6d)
#         angles_output = matrix_to_axis_angle(rotmat_recon)
#         angles_output2 = torch.reshape(angles_output, (1,-1))
#         result2, joints2, Rs2 = model(betas, angles_output2, trans)
#         verts2 = result2.squeeze().detach().cpu().numpy()
#         mv.update_mesh(vertices=verts, faces = faces,vertices2=verts2, faces2= faces)
#         print(np.round(np.sum(np.sum(verts - verts2)),2))
#         #import pdb
#         #pdb.set_trace()
#         import time
#         time.sleep(0.1)

def view_model_w_scale(opts):
    if opts.gpu != 'cpu' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SMAL(model_path='../smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl', device=device,
                 use_smal_betas=False)  # opts = opts)

    pose_size = 108
    beta_size = 9
    np.random.seed(9608)

    pose = torch.zeros((1, pose_size)).type(torch.float32).to(device)
    betas = torch.zeros((1, beta_size)).type(torch.float32).to(device)
    trans = torch.zeros((1, 3)).type(torch.float32).to(device)

    result, joints, Rs = model(betas, pose, trans, betas_logscale=torch.tensor([0.0,0,0.0,0.,0.0,1.0]))
    # limb len, limb fat, tail len, tail fat, ear y, ear z
    points = result[0].cpu().data.numpy()
    points_o = model.v_template.cpu().data.numpy()
    
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='x')
    ax.scatter(points_o[:, 0], points_o[:, 1], points_o[:, 2], c='b', marker='x')
    ax.set_xlabel('x', )
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    
if __name__ == '__main__':
    import sys, os
    # from utils.pytorch3D_angle import *

    print(sys.path)
    print(__file__)
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(base_path)
    print(sys.path)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import argparse
    parser = argparse.ArgumentParser(description='Base Processor')
    parser.add_argument('--gpu', default='cpu', help='gpu id')
    parser.add_argument('--use_smal_betas', default=False, help='if using smal shape space')
    opts = parser.parse_args()
    # test_gpu(opts)
    view_model_left_right(opts)
    # test_rotation(opts)
    # view_model_w_scale(opts)