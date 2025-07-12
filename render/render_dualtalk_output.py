from transformers import Wav2Vec2Processor
import numpy as np
import librosa
import os
import torch
import cv2
import pyrender
import trimesh

import tempfile
import imageio

from tqdm import tqdm
try:
    from psbody.mesh import Mesh
except:
    Mesh = None
import os
import torch
import numpy as np
from pytorch3d.utils import opencv_from_cameras_projection
import numpy as np
import torch
import os
import sys
import json
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from flame.FLAME_fake import FLAME, FLAMETex
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
# from utils import util
from pytorch3d.structures import Meshes

from tqdm import tqdm
from configs.config import parse_args, get_cfg_defaults, update_cfg

from util import *
import argparse
import cv2

import platform
if platform.system() == "Linux":
    # os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0, template_type: str = "flame", rgb_per_v = None):
    

    assert template_type in ["flame", "biwi"], "template_type should be one of ['flame', 'biwi'],but got {}".format(template_type)


    if template_type == "flame":
        camera_params = {'c': np.array([400, 400]),
                            'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                            'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}
    elif template_type == "biwi":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
        
    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center

    if rgb_per_v is None:
        intensity = 2.0
        primitive_material = pyrender.material.MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                    metallicFactor=0.8, 
                    roughnessFactor=0.8 
                )

        tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
    else:
        intensity = 0.5
        tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    # scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    # try:
    r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
    color, _ = r.render(scene, flags=flags)
    # except:
    #     print('pyrender: Failed rendering frame')
    #     color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_frame(args):
    predicted_vertice, f, center,  template_type = args
    render_mesh = Mesh(predicted_vertice, f)
    pred_img = render_mesh_helper(render_mesh, center, template_type=template_type)
    pred_img = pred_img.astype(np.uint8)
    return pred_img

def animate(vertices: np.array, wav_path: str, file_name: str, ply: str, fps: int = 25, vertice_gt: np.array = None, use_tqdm: bool = False, multi_process = False):
    """
    Animate the predicted vertices with the synchronized audio and save the video to the output directory.
    Args:
        vertices: (num_frames, num_vertices*3)
        wav_path: path to wav file
        file_name: name of the output file
        ply: path to the ply file
        fps: frames per second
        use_tqdm: whether to use tqdm to show the progress
        vertice_gt: (num_frames, num_vertices*3)
        template: template to use, can be "flame" or "biwi"
    """
    # make output dir
    output_dir = os.path.dirname(file_name)
    os.makedirs(output_dir, exist_ok=True)

    template = Mesh(filename=ply)
    # determine biwi or flame
    if "FLAME" in ply:
        template_type = "flame"
    elif "BIWI" in ply:
        template_type = "biwi"
    else:
        raise ValueError("Template type not recognized, please use either BIWI or FLAME")

    # reshape vertices
    predicted_vertices = vertices.reshape(-1, vertices.shape[1]//3, 3) if vertices.ndim < 3 else vertices

    num_frames = predicted_vertices.shape[0]
    if vertice_gt is not None:
        vertice_gt = vertice_gt.reshape(-1, vertice_gt.shape[1]//3, 3) if vertice_gt.ndim < 3 else vertice_gt
        num_frames = np.where(np.sum(vertice_gt, axis=(1, 2)) != 0)[0][-1] + 1 # find the number of frames where the vertices are not all zeros

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_dir)
    center = np.mean(predicted_vertices[0], axis=0)


    # make animation
    if multi_process:

        from multiprocessing import Pool, cpu_count
        from itertools import cycle
        # get maximum num of process
        frames = []
        max_processes = cpu_count()
        with Pool(processes=max_processes) as pool:
            args = [(
                predicted_vertice,
                template.f,
                center,
                template_type
            ) for predicted_vertice in predicted_vertices]

            for pred_img in pool.imap(render_frame, tqdm(args)):
                frames.append(pred_img)

        if vertice_gt is not None:
            frames_gt = []
            with Pool(processes=max_processes) as pool:
                args = [(
                    gt_vertice,
                    template.f,
                    center,
                    template_type
                ) for gt_vertice in vertice_gt]
                
                for gt_img in pool.imap(render_frame, tqdm(args)):
                    frames_gt.append(gt_img)

            # concat two videos
            frames_final = []
            for i in range(num_frames):
                frames_final.append(np.concatenate([frames_gt[i], frames[i]], axis=1))
            frames = frames_final

    else:
        frames = []
        for i_frame in tqdm(range(num_frames)) if use_tqdm else range(num_frames):
            render_mesh = Mesh(predicted_vertices[i_frame], template.f)
            pred_img = render_mesh_helper(render_mesh, center, template_type=template_type)
            pred_img = pred_img.astype(np.uint8)
            frames.append(pred_img)

        if vertice_gt is not None:
            frames_gt = []
            for i_frame in tqdm(range(num_frames)) if use_tqdm else range(num_frames):
                render_mesh = Mesh(vertice_gt[i_frame], template.f)
                pred_img = render_mesh_helper(render_mesh, center)
                pred_img = pred_img.astype(np.uint8)
                frames_gt.append(pred_img)
        
            # concat two videos
            frames_final = []
            for i in range(num_frames):
                frames_final.append(np.concatenate([frames_gt[i], frames[i]], axis=1))
            frames = frames_final

    imageio.mimsave(tmp_video_file.name, frames, fps = fps)

    cmd = " ".join(['ffmpeg', '-hide_banner -loglevel error', '-y', '-i', tmp_video_file.name, '-i', wav_path, '-c:v copy -c:a aac', '-pix_fmt yuv420p -qscale 0',file_name, ])
    cmd = " ".join(['ffmpeg', '-i', tmp_video_file.name, '-i', wav_path, '-c:v copy -c:a aac', '-pix_fmt yuv420p -qscale 0',file_name,'-y'])
    
    os.system(cmd)
    tmp_dir = tempfile.gettempdir() # check if the wav file is in the tmp dir
    if os.path.exists(wav_path) and tmp_dir in wav_path: 
        os.remove(wav_path)

    print(f"Video saved to {file_name}")


cfg = get_cfg_defaults()
config = cfg
flame = FLAME(cfg).to('cuda:0')
# mica = util.find_model_using_name(model_dir='micalib.models', model_name=config.model.name)(config, 'cuda:0')
raster_settings = RasterizationSettings(
            image_size=torch.tensor([512]),
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )
lights = PointLights(
            device='cuda:0',
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )
mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
debug_renderer = MeshRenderer(
            rasterizer=mesh_rasterizer,
            shader=SoftPhongShader(device='cuda:0', lights=lights)
        )


def render_shape(vertices, faces=None, white=True,cameras=None):
    B = vertices.shape[0]
    V = vertices.shape[1]
    if faces is None:
        faces = flame.faces
        faces = faces.cuda().repeat(B, 1, 1)
    if not white:
        verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
    else:
        verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)
    textures = TexturesVertex(verts_features=verts_rgb.cuda())
    meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

    blend = BlendParams(background_color=(1.0, 1.0, 1.0))

    fragments = mesh_rasterizer(meshes_world, cameras=cameras)
    rendering = debug_renderer.shader(fragments, meshes_world, cameras=cameras, blend_params=blend)
    rendering = rendering.permute(0, 3, 1, 2).detach()
    return rendering[:, 0:3, :, :]


output_path = "./result_DualTalk/"
audio_path = "./data/test/"


for i in os.listdir(output_path):
    if i.endswith(".npy"):
        npy_path = os.path.join(output_path, i)
        wav_path = os.path.join(audio_path, i.replace(".npy", ".wav"))
        
        file_name = npy_path.replace(".npy", "_render.mp4")
        if os.path.exists(file_name):
            print("skip", file_name)
            # continue
        npy = np.load(npy_path)
        exp = npy[:, 0:50]
        exp = np.hstack((exp, np.zeros((exp.shape[0], 50))))
        exp = torch.from_numpy(exp).float().cuda()
        jaw = npy[:, 50:53]
        # jaw = np.zeros((npy.shape[0], 3))
        jaw = torch.from_numpy(jaw).float().cuda()
        jaw = matrix_to_rotation_6d(axis_angle_to_matrix(jaw))
        neck = npy[:, 53:56]
        # neck = np.zeros((npy.shape[0], 3))
        from scipy.signal import savgol_filter
        neck = savgol_filter(neck, 7, 2, axis=0)
        neck = torch.from_numpy(neck).float().cuda()
        neck = matrix_to_rotation_6d(axis_angle_to_matrix(neck))
        gt_param = np.load(os.path.join(audio_path, i.replace(".npy", ".npz")))
        shape = gt_param["shape"]
        shape = torch.from_numpy(shape).float().cuda()
        shape = shape[0, :].unsqueeze(0)
        shape = torch.cat((shape, torch.zeros(1, 200).cuda()), dim=1)
        shape_params = shape.repeat(exp.shape[0], 1)


        vertices, _, _ = flame(
            # cameras=torch.inverse(cameras.R),
            # shape_params=None,
            shape_params=shape_params,
            expression_params=exp,
            # neck_pose_params=None,
            neck_pose_params=neck,
            eye_pose_params=None,
            jaw_pose_params=jaw,
            eyelid_params=None
        )
        exp = torch.zeros_like(exp)

        animate(vertices.cpu().numpy(), wav_path, file_name, "./render/flame/FLAME_sample.ply", fps=25, vertice_gt=None, use_tqdm=True, multi_process=True)




        if i.endswith("speaker1.npy"):
            npy_path = os.path.join(output_path, i.replace("speaker1", "speaker2"))
            wav_path = os.path.join(audio_path, i.replace("speaker1.npy", "speaker2.wav"))
            file_name = npy_path.replace(".npy", "_render.mp4")
            gt_param = np.load(os.path.join(audio_path, i.replace("speaker1.npy", "speaker2.npz")))

        else:
            npy_path = os.path.join(output_path, i.replace("speaker2", "speaker1"))
            wav_path = os.path.join(audio_path, i.replace("speaker2.npy", "speaker1.wav"))
            file_name = npy_path.replace(".npy", "_render.mp4")
            gt_param = np.load(os.path.join(audio_path, i.replace("speaker2.npy", "speaker1.npz")))
        npy = np.load(npy_path)
        exp = npy[:, 0:50]
        exp = np.hstack((exp, np.zeros((exp.shape[0], 50))))
        exp = torch.from_numpy(exp).float().cuda()
        jaw = npy[:, 50:53]
        # jaw = np.zeros((npy.shape[0], 3))
        jaw = torch.from_numpy(jaw).float().cuda()
        jaw = matrix_to_rotation_6d(axis_angle_to_matrix(jaw))
        neck = npy[:, 53:56]
        # neck = np.zeros((npy.shape[0], 3))
        from scipy.signal import savgol_filter
        neck = savgol_filter(neck, 7, 2, axis=0)
        neck = torch.from_numpy(neck).float().cuda()
        neck = matrix_to_rotation_6d(axis_angle_to_matrix(neck))
        shape = gt_param["shape"]
        shape = torch.from_numpy(shape).float().cuda()
        shape = shape[0, :].unsqueeze(0)
        shape = torch.cat((shape, torch.zeros(1, 200).cuda()), dim=1)
        shape_params = shape.repeat(exp.shape[0], 1)

        vertices, _, _ = flame(
            # cameras=torch.inverse(cameras.R),
            # shape_params=None,
            shape_params=shape_params,
            expression_params=exp,
            neck_pose_params=neck,
            eye_pose_params=None,
            jaw_pose_params=jaw,
            eyelid_params=None
        )
        # print(vertices.shape)
        animate(vertices.cpu().numpy(), wav_path, file_name, "./render/flame/FLAME_sample.ply", fps=25, vertice_gt=None, use_tqdm=True, multi_process=True)
