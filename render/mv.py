# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
sys.path.append('/home/cil/Documents/project/CrossSpecies/code')
import numpy as np
import pyrender
import trimesh
from trimesh.transformations import transform_points
import cv2

class MeshViewer(object):

    def __init__(self, width=1200, height=800,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 registered_keys=None,
                 render_flags = True, add_ground_plane = False, add_origin = False, y_up = False):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.body_color = body_color
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                                    ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0,
                                        aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        if y_up:
            camera_pose[:3,:3] = np.array([[1,0,0],[0,0,1],[0,-1,0]])
            camera_pose[:3, 3] = np.array([0, 6, 0])
        else:
            camera_pose[:3, 3] = np.array([0, 0, 1])
        self.scene.add(pc, pose=camera_pose)

        self.add_origin = add_origin
        if add_ground_plane:
            from src.utils.mv_utils import get_checkerboard_plane
            ground_mesh = pyrender.Mesh.from_trimesh(
                get_checkerboard_plane(plane_width=50, num_boxes=10, center=True),
                smooth=False,
            )
            if y_up:
                pose = trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1])
                pose[:3, 3] = [0, -0, 1.2] # (-1.3)
            else:
                pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
                pose[:3, 3] = [0, -0.8, -20]  # (-1.3)
            #pose[:3, 3] = np.array([0, body_mesh.bounds[0, 1], 0])
            self.scene.add(ground_mesh, pose=pose, name='ground_plane')

        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
                                      viewport_size=(width, height),
                                      cull_faces=False,
                                      run_in_thread=True,
                                      registered_keys=registered_keys)
                                      #render_flags={"all_wireframe":render_flags, "all_solid":False}) # visual triangle
        # Ci
        self.points_to_pymesh = pyrender.Mesh.from_points

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                    wireframe=False):
        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material)

    def get_batch_meshes(self, vertices, faces, color, wireframe=False):
        '''
        vertices list: [B,N,3] or [C][B,N,3] or [N,3]
        '''
        if len(vertices[0].shape) == 2:
            if len(color.shape) == 2:
                meshes = [self.create_mesh(vertices[i], faces, color=color[i]) for i in range(vertices.shape[0])]
            else:
                meshes = [self.create_mesh(vertices[i], faces, color = color) for i in range(vertices.shape[0])]
        elif len(vertices[0].shape) == 3:
            meshes = [self.create_mesh(vertices[i][j], faces, color=color[i]) for i in range(len(vertices)) for j in range(vertices[i].shape[0])]
        else:
            meshes = [self.create_mesh(vertices, faces, color=color)]
        return meshes

    # def get_add_textures(self, meshes, texture_images):
    #     for mesh_ in meshes:
    #         #### Insert a placeholder texture into the trimesh object to prevent "no uv_0" variable bug when defining
    #         # texture in pyrender  - the UV map used in the placeholder texture will be transferred to PyRender####
    #         # Placeholder texture
    #         texture_image = cv2.imread(texture_images)[:,:,::-1] # np.ones((1, 1, 3), dtype=np.uint8) * 255
    #         # Create a Trimesh texture
    #         texture = trimesh.visual.texture.TextureVisuals(
    #             uv=(mesh_.vertices[:, :2] - np.min(mesh_.vertices[:, :2], axis=0)) / np.ptp(mesh_.vertices[:, :2], axis=0),
    #             image=texture_image
    #         )
    #         # Set the texture for the mesh
    #         mesh_.visual = texture
    #     return meshes

    # Ci
    def create_pointcloud(self, points, colors = [0.5,0.5,0.5]):
        n = points.shape[0]
        points_colors = np.array([colors]).repeat(n, axis=0).shape
        # point_colors = np.random.uniform(size=points.shape)

        # new vertex positions
        rot = self.transf(np.radians(180), [1, 0, 0])
        new_vertices = transform_points(points,matrix=rot)

        return self.points_to_pymesh(points=new_vertices, colors=points_colors)

    # Ci
    def create_pointMesh(self, points, colors = [1.0, 0.0, 0.0]):
        n = points.shape[0]
        # new vertex positions
        rot = self.transf(np.radians(180), [1, 0, 0])
        new_vertices = transform_points(points, matrix=rot)

        sm = trimesh.creation.uv_sphere(radius=0.05)
        sm.visual.vertex_colors = colors
        tfs = np.tile(np.eye(4), (n, 1, 1))
        tfs[:, :3, 3] = new_vertices
        m = self.trimesh_to_pymesh(sm, poses=tfs)
        return m

    def update_mesh(self, vertices, faces, points = None, vertices2 = None, faces2 = None):
        '''
        Args:
            vertices: [N, 3]
            faces:  [N,3]
        Returns:
        '''
        if not self.viewer.is_active:
            return

        self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                # break
            if node.name == 'point_mesh':
                self.scene.remove_node(node)
                # break
            if node.name == 'body_mesh2':
                self.scene.remove_node(node)
                # break

        body_mesh = self.create_mesh(
            vertices, faces, color=self.body_color)
        self.scene.add(body_mesh, name='body_mesh')

        # Ci
        if self.add_origin:
            point_ori = self.create_pointMesh(points=np.array([[0,0,0]]), colors = [0.0, 0.0, 1.0])  # create_pointcloud
            self.scene.add(point_ori, name='ori_mesh')

        # Ci
        if points is not None:
            point_mesh = self.create_pointMesh(points=points, colors = [1.0, 0.0, 0.0]) #create_pointcloud
            self.scene.add(point_mesh, name='point_mesh')

        # Ci
        if vertices2 is not None:
            body_mesh2 = self.create_mesh(
                vertices2, faces, color=(0.0, 0.0, 1.0, 1.0))
            self.scene.add(body_mesh2, name='body_mesh2')
        self.viewer.render_lock.release()

    def update_multi_mesh(self, vertices, faces, points = None, color = None):
        '''
        Args:
            vertices: [B, N, 3]
            faces:  [N,3]
        Returns:
        '''
        if not self.viewer.is_active:
            return

        self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name is None:
                continue
            if 'body' in node.name:
                self.scene.remove_node(node)
                # break
            if 'point' in node.name:
                self.scene.remove_node(node)
            if node.name == 'ori_mesh':
                self.scene.remove_node(node)
                # break

        meshes = self.get_batch_meshes(vertices, faces, color)
        for i in range(len(meshes)):
            self.scene.add(meshes[i], name=f'body_mesh{i}')

        # Ci
        if self.add_origin:
            point_ori = self.create_pointMesh(points=np.array([[0, 0, 0]]),
                                              colors=[0.0, 0.0, 1.0])  # create_pointcloud
            self.scene.add(point_ori, name='ori_mesh')

        # Ci
        if points is not None:
            point_mesh = self.create_pointMesh(points=points, colors=[1.0, 0.0, 0.0])  # create_pointcloud
            self.scene.add(point_mesh, name='point_mesh')
        self.viewer.render_lock.release()