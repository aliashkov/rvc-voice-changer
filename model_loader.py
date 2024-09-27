import os
import json
import torch
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

def load_model():
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
                    
                    model_data = {
                        "name": character_name,
                        "title": model_title,
                        "author": model_author,
                        "cover": model_cover,
                        "version": "V1" if version == "v1" else "V2",
                        "tgt_sr": tgt_sr,
                        "if_f0": if_f0,
                        "version": version,
                        "index": model_index,
                    }
                    models.append(model_data)
            categories.append([category_title, category_folder, description, models])
    else:
        categories = []
    return categories

def load_model_from_checkpoint(cpt, version, if_f0):
    # Add default value for is_half
    config = cpt["config"]
    if isinstance(config, dict):
        config["is_half"] = config.get("is_half", False)
    elif isinstance(config, (list, tuple)):
        config = list(config)  # Convert tuple to list if necessary

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*config)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*config)
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*config)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*config)
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval()
    return net_g