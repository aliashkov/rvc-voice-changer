import librosa
import soundfile as sf
import traceback
import numpy as np
from datetime import datetime
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_file, abort
import librosa
import asyncio
import edge_tts
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
from rq import Queue, Worker, Connection
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq import get_current_job

#config = Config()

def perform_conversion(model_name, vc_audio_mode, vc_input, vc_upload, tts_text, tts_voice, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):
        job = get_current_job()  # Get the current job
        job.meta['progress'] = 0  # Initialize progress
        job.save_meta()  # Save initial job state  
        try:
            print("Model name: ", model_name)
            print("VC audio mode: ", vc_audio_mode)
            print("VC Input: ", vc_input)
            print("VC Upload: ", vc_upload)
            print("TTS text: ", tts_text)
            print("TTS voice: ", tts_voice)
            print("F0 UP KEY: ", f0_up_key)
            print("FO METHOD: ", f0_method)
            print("Index rate: ", index_rate)
            print("Filter rate: ", filter_radius)
            print("Resample SR: ", resample_sr)
            print("RMS MIX RATE: ", rms_mix_rate)
            print("Protect: ", protect)
            logs = []
            print(f"Converting using {model_name}...")
            logs.append(f"Converting using {model_name}...")
        except:
            info = traceback.format_exc()
            print(info)
            yield info, None
""" return vc_fn """

    #categories = load_model(config)
    #print(categories)

"""     print("Starting conversion...")

    # Load categories and models
    categories = load_model(config)
    print(categories)

    # Find the correct model
    selected_model = None
    for folder_title, folder, description, models in categories:
        for name, title, author, cover, model_version, vc_fn in models:
            if name == model_name:
                selected_model = vc_fn
                break
        if selected_model:
            break

    if not selected_model:
        return {"error": f"Model '{model_name}' not found"}
    
    vc_generator = selected_model(vc_audio_mode, vc_input, vc_upload, tts_text, tts_voice, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect)

    try:
        while True:
            logs, result = next(vc_generator)
            job.meta['progress'] += 10  # Update progress (customize this increment)
            job.save_meta()  # Save progress update
            print(logs)
    except StopIteration:
        pass

    tgt_sr, audio_opt = result
    output_path = f"converted_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    sf.write(output_path, audio_opt, tgt_sr) """

"""     job.meta['progress'] = 100  # Mark as complete
    print("65465456456")
    job.save_meta()  # Final update
    print("56564654465465465645546") """

"""     return {
        "message": f"Successfully converted using",
        "info": "777",
        "output_path": "777777"
    } """