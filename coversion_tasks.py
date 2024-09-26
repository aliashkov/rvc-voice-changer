import librosa
import soundfile as sf
import traceback
import numpy as np
from datetime import datetime
from config import Config

from converter import load_model  # Import from the new module

config = Config()

def perform_conversion(model_name, vc_audio_mode, vc_input, vc_upload, tts_text, tts_voice, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):

    print("777777777")
    # Load all categories and models
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
    
    print("5677657667567576567")

    # Call the vc_fn with all required parameters
    vc_generator = selected_model(
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

    # Get the first yield
    logs, _ = next(vc_generator)

    # Get the final result
    try:
        while True:
            logs, result = next(vc_generator)
    except StopIteration:
        pass

    if result is None:
        return {"error": logs}

    tgt_sr, audio_opt = result

    # Save the converted audio
    output_path = f"converted_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    sf.write(output_path, audio_opt, tgt_sr)

    return {
        "message": f"Successfully converted using {model_name}",
        "info": logs,
        "output_path": output_path
    }