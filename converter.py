import os
import glob
import json
import traceback
import logging
import gradio as gr
import numpy as np
from flask import Flask, request, jsonify
import librosa
import torch
import asyncio
import edge_tts
import yt_dlp
import ffmpeg
import subprocess
import sys
import io
import wave
import soundfile as sf
# import spaces
from datetime import datetime
from fairseq import checkpoint_utils
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from config import Config
import redis
from rq import Queue, Worker, Connection
from rq.job import Job
import uuid


def load_model(config):
    from app import create_vc_fn
    categories = []
    if os.path.isfile("weights/folder_info.json"):
        with open("weights/folder_info.json", "r", encoding="utf-8") as f:
            folder_info = json.load(f)
        for category_name, category_info in folder_info.items():
            if not category_info['enable']:
                continue
            category_title = category_info['title']
            category_folder = category_info['folder_path']
            description = category_info['description']
            models = []
            
            model_info_path = f"weights/{category_folder}/model_info.json"
            if os.path.isfile(model_info_path):
                with open(model_info_path, "r", encoding="utf-8") as f:
                    models_info = json.load(f)
                for character_name, info in models_info.items():
                    if not info['enable']:
                        continue
                model_title = info['title']
                model_name = info['model_path']
                model_author = info.get("author", None)
                model_cover = f"weights/{category_folder}/{character_name}/{info['cover']}"
                model_index = f"weights/{category_folder}/{character_name}/{info['feature_retrieval_library']}"
                cpt = torch.load(f"weights/{category_folder}/{character_name}/{model_name}", map_location="cpu")
                tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                if_f0 = cpt.get("f0", 1)
                version = cpt.get("version", "v1")
                if version == "v1":
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
                    model_version = "V1"
                elif version == "v2":
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
                    model_version = "V2"
                del net_g.enc_q
                print(net_g.load_state_dict(cpt["weight"], strict=False))
                net_g.eval().to(config.device)
                if config.is_half:
                    net_g = net_g.half()
                else:
                    net_g = net_g.float()
                vc = VC(tgt_sr, config)
                print(f"Model loaded: {character_name} / {info['feature_retrieval_library']} | ({model_version})")
                models.append((character_name, model_title, model_author, model_cover, model_version, create_vc_fn(model_name, tgt_sr, net_g, vc, if_f0, version, model_index)))
            categories.append([category_title, category_folder, description, models])
    else:
        categories = []
    return categories
