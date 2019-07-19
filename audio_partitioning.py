from pydub import AudioSegment
import os, sys, stat



def partitionate_audio(directory):
    seg_ms = 15    #we want to split each audio file in 20ms segments of audio
    for d in directory:
        for f in os.listdir("5Classes/"+d):
            segments = []
            audio = AudioSegment.from_wav("5Classes/"+d+"/"+f)
            len_audio = len(audio)
            i = 0
            j = 0
            while i + seg_ms < len_audio:
                new_segment = audio[i:i + seg_ms]
                segments.append(new_segment)
                i = i + seg_ms
                
            for seg in segments:
                res_path = "5Classes/New_"+str(d)
                if not os.path.isdir(res_path):
                    os.mkdir(res_path)
                    os.chmod(res_path, 0o777)
                seg.export(os.path.join(res_path, str(f).split(".")[0] + "_" + str(j)+".wav"))
                if j % 1000 == 0:
                    print("Exported: ", str(f).split(".")[0] + "_" + str(j)+".wav")
                j = j + 1



audio_directories = os.listdir("5Classes/")
audio_directories.sort()
print('Audio Classes: ', audio_directories)
print("Partitioning dataset...")
partitionate_audio(audio_directories)
