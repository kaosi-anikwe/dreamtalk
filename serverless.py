import os
import json
import uuid
import shutil
import tempfile
import requests
import traceback
import subprocess
from datetime import datetime

import torch
import runpod
import torchaudio
import numpy as np
import firebase_admin
from scipy.io import loadmat
from dotenv import load_dotenv
from firebase_admin import storage
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


load_dotenv()

# initialize firebase app
SERVICE_CERT = json.loads(os.getenv("SERVICE_CERT"))
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET")
cred_obj = firebase_admin.credentials.Certificate(SERVICE_CERT)
firebase_admin.initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET})


audio_feat_path = ""


def download_file(url):
    try:
        response = requests.get(url)
        print(f"STATUS CODE: {response.status_code}")
        response.raise_for_status()  # Raise an HTTPError for bad responses

        _, file_extension = os.path.splitext(url)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)

        with open(temp_file.name, 'wb') as file:
            file.write(response.content)

        print(f"FILENAME: {temp_file.name}")
        return temp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None


def generate_path(date=None):
    # Use the provided date or the current date if None
    if date is None:
        date = datetime.utcnow()
    # Format date components
    year = date.year
    month = date.month
    day = date.day
    # Format path
    path = os.path.join(f"{month:02d}-{year}", f"{day:02d}-{month:02d}")
    return path


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


def handler(job):
    request = job.get("input")
    audio_url = request.get("audio_url")
    wav_path = download_file(audio_url)
    image_url = request.get("image_url")
    image_path = download_file(image_url)
    if not wav_path or not image_path:
        return {
            "error": "Failed to download image or audio."
        }
    img_crop = request.get("img_crop", True)
    style_clip_path = request.get("style_clip_path", "data/style_clip/3DMM/M030_front_neutral_level1_001.mat")
    pose_path = request.get("pose_path", "data/pose/RichardShelby_front_neutral_level1_001.mat")
    max_gen_len = request.get("max_gen_len", 30)
    cfg_scale = request.get("cfg_scale", 1.0)
    output_name = uuid.uuid4().hex
    device = request.get("device", "cuda")

    try:
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

        global audio_feat_path
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

            storage_client = storage.bucket()
            path = os.path.join("dreamtalk", generate_path(), f"{output_name}.mp4")
            # upload video
            blob = storage_client.blob(path)
            blob.upload_from_filename(output_video_path)
            blob.make_public()
            blob_url = blob.public_url

            os.remove(output_video_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(image_path):
                os.remove(image_path)

            return {
                "videoURL": blob_url
            }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        if os.path.exists(f"tmp/{output_name}"):
            shutil.rmtree(f"tmp/{output_name}")

runpod.serverless.start({"handler": handler})
