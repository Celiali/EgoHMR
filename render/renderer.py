# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2


class COLORRenderer(object):

    def __init__(self, focal_length=5000, img_w=256, img_h=256, faces=None,
                 same_mesh_color=True):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        self.right_index = range(813, 1497)
        self.left_index = range(0, 813)
        self.right_faces = range(1, 2991, 2)
        self.left_faces = range(0, 2990, 2)

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        '''
        Args:
            verts: [B, N , 3]
            bg_img_rgb:
            bg_color:
        Returns:
        '''
        # verts[:,:,2]+=5
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        # camera = pyrender.camera.OrthographicCamera(xmag= 1, ymag = 1.0)
        pose = np.eye(4)
        pose[0,3] = -0.5
        scene.add(camera, pose=pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        rot2 = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        # multiple person
        num_people = verts.shape[0] #len(verts)
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            rot2[2, 3] = -80
            # rot2[2, 0] = 40
            mesh.apply_transform(rot2)
            if self.same_mesh_color:
                mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
            else:
                mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            # mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            colors_faces = np.zeros_like(mesh.faces)
            colors_faces[self.left_faces, :] = np.array([0.654, 0.396, 0.164])*255.
            colors_faces[self.right_faces, :] = np.array([.7, .7, .7])*255.
            mesh.visual.face_colors = colors_faces  # np.random.uniform(size=mesh.faces.shape)
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        return color_rgb

    def render_side_view(self, verts, bg_color=(0, 0, 0, 0), obtainSil = False, type = 'IMG'):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(verts = pred_vert_arr_side,  bg_color=bg_color ,obtainSil=obtainSil, type = type)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()