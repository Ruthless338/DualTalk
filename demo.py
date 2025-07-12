#-*- coding : utf-8-*-

import argparse
import os
import platform
import subprocess
import tempfile
import time

import cv2
import imageio
import librosa
import numpy as np
import pyrender
import torch
import trimesh
from pytorch3d.renderer import (
    BlendParams, MeshRasterizer, MeshRenderer, PointLights,
    RasterizationSettings, SoftPhongShader, TexturesVertex
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from render.configs.config import get_cfg_defaults
from render.flame.FLAME_fake import FLAME
from render.util import *
from scipy.signal import savgol_filter
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from dataset_test import get_loader
from DualTalk import DualTalkModel

try:
    from psbody.mesh import Mesh
except:
    Mesh = None

if platform.system() == "Linux":
    # os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# export HF_ENDPOINT=https://hf-mirror.com
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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


@torch.no_grad()
def test(args, model):
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    model.load_state_dict(torch.load(os.path.join(args.save_path, "DualTalk.pth"),map_location=torch.device('cpu')))
    model = model.to(args.device)
    model.eval()
    # for file_name, audio1, audio2, exp2, jawpose2, neck2 in test_loader:
    audio1_path = args.audio1_path
    audio2_path = args.audio2_path
    bs2_path = args.bs2_path
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    speech_array, sampling_rate = librosa.load(audio1_path, sr=16000)
    audio1 = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio1 = torch.from_numpy(audio1).float().to(args.device)
    audio1 = audio1.unsqueeze(0)
    speech_array, sampling_rate = librosa.load(audio2_path, sr=16000)
    audio2 = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio2 = torch.from_numpy(audio2).float().to(args.device)
    audio2 = audio2.unsqueeze(0)
    exp2 = torch.from_numpy(np.load(bs2_path)['exp']).float().to(args.device)
    jawpose2 = torch.from_numpy(np.load(bs2_path)['pose'][:,3:]).float().to(args.device)
    neck2 = torch.from_numpy(np.load(bs2_path)['pose'][:,:3]).float().to(args.device)
    blendshape2 = torch.cat((exp2, jawpose2, neck2), dim=1).unsqueeze(0)
    #计算时间
    start_time = time.time()
    prediction = model(audio1,audio2,blendshape2)
    end_time = time.time()
    frames = prediction.shape[1]
    file_name = audio1_path.split('/')[-1].replace('.wav', '')
    prediction = prediction.squeeze().detach().cpu().numpy()
    np.save(os.path.join(result_path, "{}.npy".format( file_name)), prediction)
    #计算平均fps



def main():
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    parser = argparse.ArgumentParser(description='DualTalk')
    parser.add_argument("--audio1_path", type=str, default= "./demo/xkHwlcDSOjc_sub_video_109_000_speaker2.wav", help='path of the test data')
    parser.add_argument("--audio2_path", type=str, default= "./demo/xkHwlcDSOjc_sub_video_109_000_speaker1.wav", help='path of the test data')
    parser.add_argument("--bs2_path", type=str, default= "./demo/xkHwlcDSOjc_sub_video_109_000_speaker1.npz", help='path of the test data')
    parser.add_argument("--seed", type=int, default=6666, help='random seed')
    parser.add_argument("--scale",type=str, default="large",help="large or base")
    parser.add_argument("--blendshape_dim", type=int, default=56, help='number of blendshapes:52')
    parser.add_argument("--feature_dim", type=int, default=256, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--device", type=str, default="cuda:0", help='cuda:0 or cuda:1')
    parser.add_argument("--save_path", type=str, default="./model/", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="./result_DualTalk", help='path to the predictions')
    parser.add_argument("--max_seq_len", type=int, default=5000, help='max_seq_len')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    model = DualTalkModel(args)
    model = model.to(args.device)
    test(args, model)
    ply_path = "./render/flame/FLAME_sample.ply"
    npz_path = args.bs2_path
    wav_path = args.audio2_path
    save_path = args.result_path
    data_name = args.audio2_path.split('/')[-1].replace('.wav', '')
    data = np.load(npz_path)
    shape = data["shape"][0,:]
    shape = shape.reshape(1,-1)
    exp = data["exp"]
    exp = savgol_filter(exp, 7, 3, axis=0)

    shape = np.hstack((shape, np.zeros((shape.shape[0], 200))))
    shape = shape.repeat(exp.shape[0], axis=0)
    shape = torch.from_numpy(shape).float().cuda()#.unsqueeze(0)
    exp = np.hstack((exp, np.zeros((exp.shape[0], 50))))
    exp = torch.from_numpy(exp).float().cuda()#.unsqueeze(0)
    jaw = data["pose"][:,3:]
    jaw = torch.from_numpy(jaw).float().cuda()#.unsqueeze(0)
    jaw = matrix_to_rotation_6d(axis_angle_to_matrix(jaw))
    neck = data["pose"][:,:3]
    neck = savgol_filter(neck, 7, 3, axis=0)
    neck = torch.from_numpy(neck).float().cuda()#.unsqueeze(0)
    neck = matrix_to_rotation_6d(axis_angle_to_matrix(neck))
    file_name1 = os.path.join(save_path, "{}".format(data_name)+"_render.mp4")
    vertices, _, _ = flame(
    shape_params=shape,
    expression_params=exp,
    neck_pose_params=neck,
    eye_pose_params=None,
    jaw_pose_params=jaw,
    eyelid_params=None
    )
    animate(vertices.cpu().numpy(), wav_path, file_name1, ply_path, fps=25, vertice_gt=None, use_tqdm=True, multi_process=True)

    data_name = args.audio1_path.split('/')[-1].replace('.wav', '')
    prediction = np.load(os.path.join(args.result_path, "{}.npy".format(data_name)))
    wav_path = args.audio1_path
    file_name2 = os.path.join(save_path, "{}".format(data_name)+"_render.mp4")
    exp = prediction[:, 0:50]
    exp = np.hstack((exp, np.zeros((exp.shape[0], 50))))
    exp = torch.from_numpy(exp).float().cuda()
    jaw = prediction[:, 50:53]
    # jaw = np.zeros((npy.shape[0], 3))
    jaw = torch.from_numpy(jaw).float().cuda()
    jaw = matrix_to_rotation_6d(axis_angle_to_matrix(jaw))
    neck = prediction[:, 53:56]
    # neck = np.zeros((npy.shape[0], 3))
    neck = savgol_filter(neck, 7, 2, axis=0)
    neck = torch.from_numpy(neck).float().cuda()
    neck = matrix_to_rotation_6d(axis_angle_to_matrix(neck))
    vertices, _, _ = flame(
            # cameras=torch.inverse(cameras.R),
            shape_params=None,
            # shape_params=shape_params,
            expression_params=exp,
            # neck_pose_params=None,
            neck_pose_params=neck,
            eye_pose_params=None,
            jaw_pose_params=jaw,
            eyelid_params=None
        )
    animate(vertices.cpu().numpy(), wav_path, file_name2, ply_path, fps=25, vertice_gt=None, use_tqdm=True, multi_process=True)
    video1_path = file_name1
    video2_path = file_name2
    output_video = file_name2.replace("_render.mp4", ".mp4")
    ffmpeg_command = [
        "ffmpeg",
        "-i", video1_path,
        "-i", video2_path,
        "-filter_complex", "[0:v]pad=iw*2:ih[left];[left][1:v]overlay=w[video];[0:a][1:a]amix=inputs=2[audio]",
        "-map", "[video]",
        "-map", "[audio]",
        output_video,
        "-y"
    ]
    subprocess.run(ffmpeg_command, check=True)




if __name__ == "__main__":
    main()