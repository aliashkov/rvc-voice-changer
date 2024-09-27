import librosa
import soundfile as sf
import traceback
import numpy as np
from datetime import datetime
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_file, abort
import librosa
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
from test import calculate_stats
#from converter import load_model

def perform_conversion(model_name, vc_audio_mode, vc_input, vc_upload, tts_text, tts_voice, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, categories ):
        job = get_current_job()  # Get the current job
        job.meta['progress'] = 0  # Initialize progress
        job.save_meta()  # Save initial job state  
        test_array = [1, 2, 3, 4, 5]
        mean, std_dev = calculate_stats(test_array)
        print(f"Mean: {mean}, Standard Deviation: {std_dev}")
        try:
            #categories = load_model(config)
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
            print(f"Categories:  {categories}")
            logs = []
            print(f"Converting using {model_name}...")
            logs.append(f"Converting using {model_name}...")
            if vc_audio_mode == "Upload audio":
                if vc_upload is None:
                   print(f"You need to upload an audio")
                   logs.append(f"You need to upload an audio")
                   yield "\n".join(logs), None
                   return "You need to upload an audio", None
                sampling_rate, audio = vc_upload
                print("Audio", audio)
                print("Sampling_rate", sampling_rate)
                duration = audio.shape[0] / sampling_rate
                print("Duration", duration)
                if duration > 20:
                    print(f"Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.")
                    logs.append(f"Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.")
                    yield "\n".join(logs), None
                    return "Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.", None
                print("Audio_type ", audio.dtype)
                print(audio.dtype != np.float32)
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