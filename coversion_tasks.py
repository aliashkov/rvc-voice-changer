import librosa
import soundfile as sf
import traceback
import numpy as np
from datetime import datetime
from config import Config
from rq import get_current_job

from converter import load_model  # Import from the new module

config = Config()

def perform_conversion(model_name, vc_audio_mode, vc_input, vc_upload, tts_text, tts_voice, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):
    job = get_current_job()  # Get the current job
    job.meta['progress'] = 0  # Initialize progress
    job.save_meta()  # Save initial job state

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

    job.meta['progress'] = 100  # Mark as complete
    job.save_meta()  # Final update

    return {
        "message": f"Successfully converted using {model_name}",
        "info": "777",
        "output_path": "777777"
    }