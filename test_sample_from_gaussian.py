import numpy as np
import pickle as pkl
from models.smal import HSMAL
import torch
from tqdm import tqdm
prior_path = '/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smpl_models/walking_toy_symmetric_smal_0000_new_skeleton_pose_prior_new_36parts.pkl'

with open(prior_path, "rb") as f:
    res = pkl.load(f, encoding='latin1')
mean_ch = res['mean_pose']
precs_ch = res['pic']
# Try sampling to double check.
cov = np.linalg.inv(precs_ch.dot(precs_ch.T))

num_samples = 60

################################# load smpl models
# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
smal = HSMAL(
    model_path='/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl',
    model_type='hsmal',
    num_betas=9).to(device)
################################### create meshviewer
from render.mv import MeshViewer

mv = MeshViewer(width=1200, height=800,
                body_color=(1.0, 1.0, 0.9, 1.0),
                registered_keys=None,
                render_flags=True, add_ground_plane=False, add_origin=False, y_up=True)

xx, yy, zz = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 6), [0])
grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
trans = grid * 10.
xx, yy, zz = np.meshgrid(np.linspace(0, 2, 10), np.linspace(0, 2, 6), [0])
grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
colors = (grid * 255).astype(int)

for i in tqdm(range(100)):
    # Generate samples
    samples = np.random.multivariate_normal(mean_ch, cov, num_samples)
    ###### get pred smpl joints / vertices
    pred_output = smal(body_pose=torch.from_numpy(samples).float().to(device),
                       global_orient=torch.zeros((num_samples, 3)).float().to(device), pose2rot=True)
    pred_vertices = pred_output.vertices
    pred_vertices_full_numpy = pred_vertices.cpu().data.numpy()  # [bs, nsample, 1497,3 ]
    translated_verts = pred_vertices_full_numpy.reshape(-1, 1497, 3) + np.expand_dims(trans, axis=1)

    mv.update_multi_mesh(vertices=translated_verts, faces=smal.faces, points=None, color=colors)