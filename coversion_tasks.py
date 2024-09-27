import os
import torch
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import traceback
from model_loader import load_model
from vc_infer_pipeline import VC
from fairseq import checkpoint_utils
from config import Config
from rq import get_current_job


config = Config()

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()



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

        print("Audio mode", vc_audio_mode)
        
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

        print("File index", file_index)
        
        """ audio_opt = vc.pipeline(
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
        ) """

        times = [0, 0, 0]
        f0_up_key = int(f0_up_key)

        audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                vc_input,
                times,
                f0_up_key,
                f0_method,
                file_index,
                # file_big_npy,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                f0_file=None,
            )
        
        return logs, (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, None

def perform_conversion(model_name, vc_audio_mode, vc_input, vc_upload, tts_text, tts_voice, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):
    job = get_current_job()

    # Set progress to 0
    job.meta['progress'] = 0
    job.save_meta()

    # Load model
    categories = load_model()
    job.meta['progress'] = 10
    job.save_meta()


    load_hubert()
    job.meta['progress'] = 15
    job.save_meta()

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
    
    job.meta['progress'] = 20
    job.save_meta()
    
    print(folder_title, folder, description, models)
    
    #cpt = torch.load(f"weights/{folder}/{selected_model['name']}/{selected_model['name']}.pth", map_location="cpu")

    # print("CPT ", cpt)
    print("CATEGORIES" , categories)
    print("SELECTED MODEL" , selected_model)

    # character_name, model_title, model_author, model_cover, model_version = selected_model
    tgt_sr = selected_model["tgt_sr"]
    vc = VC(tgt_sr, config)

    job.meta['progress'] = 30
    job.save_meta()

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

    job.meta['progress'] = 90
    job.save_meta()

    if audio_result is None:
        return {"error": logs}

    tgt_sr, audio_opt = audio_result

    output_path = f"converted_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    sf.write(output_path, audio_opt, tgt_sr)

    job.meta['progress'] = 100
    job.save_meta()

    return {
        "message": f"Successfully converted using {model_name}",
        "info": logs,
        "output_path": output_path
    }