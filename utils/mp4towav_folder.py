import os
import pdb
import noisereduce as nr

from tqdm import tqdm
from os import path
from pydub import AudioSegment
from scipy.io import wavfile


# files
src_folder = "/home/nttung/research/Monash_CCU/mock_data/visual_data/format_mp4_video"
dst_folder = "/home/nttung/research/Monash_CCU/mock_data/audio_data"

for each_file in tqdm(os.listdir(src_folder)):
    print(each_file)
    src_full_path = os.path.join(src_folder, each_file)
    dst_full_path = os.path.join(dst_folder, each_file.split('.')[0]+'.wav')

    # convert mp4 to wav
    sound = AudioSegment.from_file(src_full_path,format="mp4")
    sound.export(dst_full_path, format="wav")

    rate, data = wavfile.read(dst_full_path)
    # # perform noise reduction
    # reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(dst_full_path, rate, data)