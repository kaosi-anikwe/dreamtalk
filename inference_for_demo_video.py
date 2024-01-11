import argparse
import json
import os
import uuid
import shutil
import subprocess

import numpy as np
import torch
import torchaudio
from scipy.io import loadmat
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from configs.default import get_cfg_defaults
from core.networks.diffusion_net import DiffusionNet
from core.networks.diffusion_util import NoisePredictor, VarianceSchedule
from core.utils import (
    crop_src_image,
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
)
from generators.utils import get_netG, render_video


audio_feat_path = ""


@torch.no_grad()
def get_diff_net(cfg, device):
    diff_net = DiffusionNet(
        cfg=cfg,
        net=NoisePredictor(cfg),
        var_sched=VarianceSchedule(
            num_steps=cfg.DIFFUSION.SCHEDULE.NUM_STEPS,
            beta_1=cfg.DIFFUSION.SCHEDULE.BETA_1,
            beta_T=cfg.DIFFUSION.SCHEDULE.BETA_T,
            mode=cfg.DIFFUSION.SCHEDULE.MODE,
        ),
    )
    checkpoint = torch.load(cfg.INFERENCE.CHECKPOINT, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    diff_net_dict = {
        k[9:]: v for k, v in model_state_dict.items() if k[:9] == "diff_net."
    }
    diff_net.load_state_dict(diff_net_dict, strict=True)
    diff_net.eval()

    return diff_net


@torch.no_grad()
def get_audio_feat(wav_path, output_name, wav2vec_model):
    audio_feat_dir = os.path.dirname(audio_feat_path)

    pass


@torch.no_grad()
def inference_one_video(
    cfg,
    audio_path,
    style_clip_path,
    pose_path,
    output_path,
    diff_net,
    device,
    max_audio_len=None,
    sample_method="ddim",
    ddim_num_step=10,
):
    audio_raw = audio_data = np.load(audio_path)

    if max_audio_len is not None:
        audio_raw = audio_raw[: max_audio_len * 50]
    gen_num_frames = len(audio_raw) // 2

    audio_win_array = get_wav2vec_audio_window(
        audio_raw,
        start_idx=0,
        num_frames=gen_num_frames,
        win_size=cfg.WIN_SIZE,
    )

    audio_win = torch.tensor(audio_win_array).to(device)
    audio = audio_win.unsqueeze(0)

    # the second parameter is "" because of bad interface design...
    style_clip_raw, style_pad_mask_raw = get_video_style_clip(
        style_clip_path, "", style_max_len=256, start_idx=0
    )

    style_clip = style_clip_raw.unsqueeze(0).to(device)
    style_pad_mask = (
        style_pad_mask_raw.unsqueeze(0).to(device)
        if style_pad_mask_raw is not None
        else None
    )

    gen_exp_stack = diff_net.sample(
        audio,
        style_clip,
        style_pad_mask,
        output_dim=cfg.DATASET.FACE3D_DIM,
        use_cf_guidance=cfg.CF_GUIDANCE.INFERENCE,
        cfg_scale=cfg.CF_GUIDANCE.SCALE,
        sample_method=sample_method,
        ddim_num_step=ddim_num_step,
    )
    gen_exp = gen_exp_stack[0].cpu().numpy()

    pose_ext = pose_path[-3:]
    pose = None
    pose = get_pose_params(pose_path)
    # (L, 9)

    selected_pose = None
    if len(pose) >= len(gen_exp):
        selected_pose = pose[: len(gen_exp)]
    else:
        selected_pose = pose[-1].unsqueeze(0).repeat(len(gen_exp), 1)
        selected_pose[: len(pose)] = pose

    gen_exp_pose = np.concatenate((gen_exp, selected_pose), axis=1)
    np.save(output_path, gen_exp_pose)
    return output_path


def handler(request):
    params = request.get("input")
    audio_url = params.get("audio_url")
    wav_path = "" # TODO: download audio
    image_url = params.get("image_url")
    image_path = "" # TODO: download image
    img_crop = params.get("img_crop", True)
    style_clip_path = params.get("style_clip_path", "")
    pose_path = params.get("pose_path", "")
    max_gen_len = params.get("max_gen_len", 1000)
    cfg_scale = params.get("cfg_scale", 1.0)
    output_name = uuid.uuid4().hex
    device = params.get("device", "cuda")


    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, set --device=cpu to use CPU.")
        exit(1)

    device = torch.device(device)

    cfg = get_cfg_defaults()
    cfg.CF_GUIDANCE.SCALE = cfg_scale
    cfg.freeze()

    tmp_dir = f"tmp/{output_name}"
    os.makedirs(tmp_dir, exist_ok=True)

    # get audio in 16000Hz
    wav_16k_path = os.path.join(tmp_dir, f"{output_name}_16K.wav")
    command = f"ffmpeg -y -i {wav_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_16k_path}"
    subprocess.run(command.split())

    # get wav2vec feat from audio
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    )

    wav2vec_model = (
        Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        .eval()
        .to(device)
    )

    speech_array, sampling_rate = torchaudio.load(wav_16k_path)
    audio_data = speech_array.squeeze().numpy()
    inputs = wav2vec_processor(
        audio_data, sampling_rate=16_000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        audio_embedding = wav2vec_model(
            inputs.input_values.to(device), return_dict=False
        )[0]

    audio_feat_path = os.path.join(tmp_dir, f"{output_name}_wav2vec.npy")
    np.save(audio_feat_path, audio_embedding[0].cpu().numpy())

    # get src image
    src_img_path = os.path.join(tmp_dir, "src_img.png")
    if img_crop:
        crop_src_image(image_path, src_img_path, 0.4)
    else:
        shutil.copy(image_path, src_img_path)

    with torch.no_grad():
        # get diff model and load checkpoint
        diff_net = get_diff_net(cfg, device).to(device)
        # generate face motion
        face_motion_path = os.path.join(tmp_dir, f"{output_name}_facemotion.npy")
        inference_one_video(
            cfg,
            audio_feat_path,
            style_clip_path,
            pose_path,
            face_motion_path,
            diff_net,
            device,
            max_audio_len=max_gen_len,
        )
        # get renderer
        renderer = get_netG("checkpoints/renderer.pt", device)
        # render video
        output_video_path = f"output_video/{output_name}.mp4"
        render_video(
            renderer,
            src_img_path,
            face_motion_path,
            wav_16k_path,
            output_video_path,
            device,
            fps=25,
            no_move=False,
        )
