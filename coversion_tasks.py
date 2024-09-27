import os
import torch
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import traceback
from model_loader import load_model
from vc_infer_pipeline import VC
from config import Config

config = Config()


def vc_fn(
    model_name,
    tgt_sr,
    net_g,
    vc,
    if_f0,
    version,
    file_index,
    vc_audio_mode,
    vc_input, 
    vc_upload,
    tts_text,
    tts_voice,
    f0_up_key,
    f0_method,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):
    try:
        logs = []
        print(f"Converting using {model_name}...")
        logs.append(f"Converting using {model_name}...")
        
        if vc_audio_mode == "Upload audio":
            if vc_upload is None:
                return "You need to upload an audio", None
            sampling_rate, audio = vc_upload
            duration = audio.shape[0] / sampling_rate
            if duration > 20:
                return "Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.", None
            if audio.dtype != np.float32:
                if np.issubdtype(audio.dtype, np.integer):
                    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                else:
                    audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.transpose(1, 0))
            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        else:
            return "Invalid audio mode", None

        f0_up_key = int(f0_up_key)
        
        audio_opt = vc.pipeline(
            net_g,
            audio,
            sid=0,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            file_index=file_index,
            index_rate=index_rate,
            if_f0=if_f0,
            filter_radius=filter_radius,
            tgt_sr=tgt_sr,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )
        
        return logs, (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, None

def perform_conversion(model_name, vc_audio_mode, vc_input, vc_upload, tts_text, tts_voice, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):
    categories = load_model()

    print("CATEGORIES", categories)

    selected_model = None
    for folder_title, folder, description, models in categories:
        for model_data in models:
            if model_data["name"] == model_name:
                selected_model = model_data
                break
        if selected_model:
            break

    if not selected_model:
        return {"error": f"Model '{model_name}' not found"}
    
    print(folder_title, folder, description, models)
    
    #cpt = torch.load(f"weights/{folder}/{selected_model['name']}/{selected_model['name']}.pth", map_location="cpu")

    # print("CPT ", cpt)
    print("CATEGORIES" , categories)
    print("SELECTED MODEL" , selected_model)
    #vc = VC(tgt_sr, config)

    result = vc_fn(
        model_name,
        selected_model["tgt_sr"],
        selected_model["net_g"],
        vc,
        selected_model["if_f0"],
        selected_model["version"],
        selected_model["index"],
        vc_audio_mode,
        vc_input,
        vc_upload,
        tts_text,
        tts_voice,
        f0_up_key,
        f0_method,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect
    )

    logs, audio_result = result

    if audio_result is None:
        return {"error": logs}

    tgt_sr, audio_opt = audio_result

    output_path = f"converted_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    sf.write(output_path, audio_opt, tgt_sr)

    return {
        "message": f"Successfully converted using {model_name}",
        "info": logs,
        "output_path": output_path
    }