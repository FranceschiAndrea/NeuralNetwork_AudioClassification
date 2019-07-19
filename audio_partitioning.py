from pydub import AudioSegment
import os, sys, stat



def partitionate_audio(parent_dir, new_parent_dir, seg_ms, training_path, testing_path):
    audio_directories = os.listdir(parent_dir)
    audio_directories.sort()
    print('Audio Classes: ', audio_directories)
    print("Partitioning dataset...")
    
    for d in audio_directories:
        for f in os.listdir(parent_dir + "/"+d):
            segments = []
            audio = AudioSegment.from_wav(parent_dir + "/"  + d + "/" + f)
            len_audio = len(audio)
            i = 0
            j = 0
            while i + seg_ms < len_audio:
                new_segment = audio[i:i + seg_ms]
                segments.append(new_segment)
                i = i + seg_ms
            train_segm_len = int(0.75*len(segments))
            test_segm_len = int(0.25*len(segments))
            for k in range(train_segm_len):
                res_path = training_path + "/" +  str(d)
                if not os.path.isdir(res_path):
                    os.mkdir(res_path)
                    os.chmod(res_path, 0o777)
                seg = segments[k]
                seg.export(os.path.join(res_path, str(f).split(".")[0] + "_" + str(j)+".wav"))
                if j % 1000 == 0:
                    print("Exported: ", str(f).split(".")[0] + "_" + str(j)+".wav")
                j = j + 1
                
            for k in range(test_segm_len):
                res_path = testing_path + "/" + str(d)
                if not os.path.isdir(res_path):
                    os.mkdir(res_path)
                    os.chmod(res_path, 0o777)
                seg = segments[k]
                seg.export(os.path.join(res_path, str(f).split(".")[0] + "_" + str(j)+".wav"))
                if j % 1000 == 0:
                    print("Exported: ", str(f).split(".")[0] + "_" + str(j)+".wav")
                j = j + 1




