import os
import pdb
import sys
from tqdm import tqdm

# text

# audio
import wave
import json
import os.path as osp
import whisper
import librosa
from pyannote.audio import Pipeline
from googletrans import Translator


# from pydub import AudioSegment
from io import BytesIO

def extract_transcript(audio_folder_path, diar_res):

    # Translation model
    translator = Translator()

    # Transcription model
    transcript_model = whisper.load_model("large")

    options = whisper.DecodingOptions()

    # Diarization model
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token="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")

    for audio_name in tqdm(diar_res):
        print(audio_name)
        file_path = osp.join(audio_folder_path, audio_name+'.wav')

        _, sr = librosa.load(file_path)
        audio = whisper.load_audio(file_path, sr)
        diarization = pipeline(file_path)

        audio_diar = diar_res[audio_name]["all_D"]
        # add one more information
        diar_res[audio_name]["all_transcription_en"] = []
        diar_res[audio_name]["all_transcription_zh"] = []

        for [start, end] in tqdm(audio_diar):
            
            utt_segment = audio[int(start*sr):int(end*sr)]
            utt_segment = whisper.pad_or_trim(utt_segment)

            mel = whisper.log_mel_spectrogram(utt_segment).to(transcript_model.device)

            # text
            text = whisper.decode(transcript_model, mel, options).text

            # check language
            _, probs = transcript_model.detect_language(mel)
            if max(probs, key=probs.get) == 'en':
                en = text
            else:
                en = translator.translate(text).text
            diar_res[audio_name]['all_transcription_en'].append(en)
            diar_res[audio_name]['all_transcription_zh'].append(text)
        
    # write
    with open('/home/nttung/research/Monash_CCU/CL_Monash/auxiliary_folder/SD_diar_result_with_trans.json', 'w') as fp:
        json.dump(diar_res, fp, indent=4)


if __name__ == "__main__":
    audio_folder_path = '/home/nttung/research/Monash_CCU/mock_data/audio_data'

    with open('/home/nttung/research/Monash_CCU/CL_Monash/auxiliary_folder/SD_diar_result.json', 'r') as fp:
        diar_res = json.load(fp)

    extract_transcript(audio_folder_path, diar_res)


    


    