import numpy as np
import smplx
import torch
import trimesh

class SMAL(smplx.SMPL):
    NUM_JOINTS = 34
    NUM_BODY_JOINTS = 34
    SHAPE_SPACE_DIM = 41

    def __init__(self, model_path, model_type='smal', num_betas=41, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertex_joint_selector.extra_joints_idxs = torch.empty(0, dtype=torch.int32)

class HSMAL(smplx.SMPL):
    NUM_JOINTS = 35
    NUM_BODY_JOINTS = 35
    SHAPE_SPACE_DIM = 1369

    def __init__(self, model_type='hsmal', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type=model_type
        self.vertex_joint_selector.extra_joints_idxs = torch.empty(0, dtype=torch.int32)

if __name__ == '__main__':
    smal = HSMAL(model_path='/home/cil/Documents/project/code_from_others/Smplx/EgoHMR/render/SMAL/smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl',
                 model_type='hsmal',
                 num_betas=9)
    faces = smal.faces
    print("ok")
    import torch
    pose = torch.from_numpy((np.random.rand(32, 108) - 0.5) * 0.4).type(torch.float32)
    betas = torch.from_numpy((np.random.rand(32, 9) - 0.5) * 0.06).type(torch.float32)
    trans = torch.from_numpy(np.zeros((32, 3))).type(torch.float32)
    pred_smpl_params = {'betas': betas, 'trans': trans,
                        'global_orient': pose[:,:3],
                        'body_pose': pose[:,3:]}
    smal_output = smal(**{k: v.float() for k, v in pred_smpl_params.items()}, return_full_pose=True,
                                 pose2rot=True)
    vertices = smal_output.vertices  # [bs, 6890, 3]

    import sys
    sys.path.append('/home/cil/Documents/project/code_from_others/Smplx/EgoHMR')
    from render.mv import MeshViewer
    mv = MeshViewer(width=1200, height=800,
                    body_color=(1.0, 1.0, 0.9, 1.0),
                    registered_keys=None,
                    render_flags=True, add_ground_plane=False, add_origin=False, y_up=False)
    mv.update_multi_mesh(vertices=vertices.cpu().data.numpy(), faces=faces, points=None)

