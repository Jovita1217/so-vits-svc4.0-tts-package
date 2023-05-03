import io
import os
import re
import torch
#import argparse
from collections import OrderedDict
import glob
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile as sf
from inference.infer_tool import Svc
from inference.infer_tool_768l12 import Svc_768l12
import logging
import json
import matplotlib.pyplot as plt
import parselmouth
import time
import subprocess
import shutil
import asyncio
import utils
import datetime
import traceback
from utils import mix_model
from onnxexport.model_onnx import SynthesizerTrn
import edge_tts

from scipy.io import wavfile

#parser = argparse.ArgumentParser()
#parser.add_argument("--user", type=str, help='set gradio user', default=None)
#parser.add_argument("--password", type=str, help='set gradio password', default=None)
#cmd_opts = parser.parse_args()

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

raw_path = "./dataset_raw"
raw_wavs_path = "./raw"
models_backup_path = './models_backup'
root_dir = "checkpoints"
debug = False
#now_dir = os.getcwd()

def debug_change():
    global debug
    debug = debug_button.value

def get_model_info(choice_ckpt):
    pthfile = "logs/44k/" + choice_ckpt
    if torch.cuda.is_available() == True:
        net = torch.load(pthfile)
    else:
        net = torch.load(pthfile,map_location=torch.device('cpu'))
    first_key = next(iter(net["model"]))
    if first_key == "emb_g.weight":
        value_size = net["model"][first_key].size()
    else:
        return "所选模型缺少emb_g.weight，你可能选择了一个底模"
        #return gr.Textbox.update(value="所选模型缺少emb_g.weight，你可能选择了一个底模", interactive=False)
    layer = str(value_size).split(",")[1].strip().strip("]")
    if layer[0:3] == "768":
        return "Vec768-Layer12"
        #return gr.Textbox.update(value="Vec768-Layer12", interactive=False)
    elif layer[0:3] == "256":
        return "v1"
        #return gr.Textbox.update(value="v1", interactive=False)
    else:
        return "不受支持的模型"
        #return gr.Textbox.update(value="不受支持的模型", interactive=False)
    
def check_json(config_choice, model_branch):
    model_channels = None
    if model_branch == "v1":
        model_channels = "256"
    elif model_branch == "Vec768-Layer12":
        model_channels = "768"
    else:
        return gr.Textbox.update("请先选择模型")
    config_file = "configs/" + config_choice
    with open(config_file, 'r') as f:
        config = json.load(f)
    config_channels = str(config["model"]["gin_channels"])
    if model_channels != "256" and model_channels != "768":
        return gr.Textbox.update(value="奇怪的配置文件")
    elif config_channels == model_channels:
        return gr.Textbox.update(value="模型与配置文件匹配成功")
    else:
        return gr.Textbox.update(value="模型与配置文件不匹配，请检查模型分支，否则推理可能出错")


def load_model_func(ckpt_name,cluster_name,config_name,enhance,model_branch):
    global model, cluster_model_path

    config_path = "configs/" + config_name

    with open(config_path, 'r') as f:
        config = json.load(f)
    spk_dict = config["spk"]
    spk_name = config.get('spk', None)
    if spk_name:
        spk_choice = next(iter(spk_name))
    else:
        spk_choice = "未检测到音色"

    ckpt_path = "logs/44k/" + ckpt_name
    cluster_path = "logs/44k/" + cluster_name
    if cluster_name == "no_clu" and model_branch == "v1":
            model = Svc(ckpt_path,config_path,nsf_hifigan_enhance=enhance)
    elif cluster_name == "no_clu" and model_branch == "Vec768-Layer12":
            model = Svc_768l12(ckpt_path,config_path,nsf_hifigan_enhance=enhance)
    elif cluster_name != "no_clu" and model_branch == "v1":
            model = Svc(ckpt_path,config_path,cluster_model_path=cluster_path,nsf_hifigan_enhance=enhance)
    else:
            model = Svc_768l12(ckpt_path,config_path,cluster_model_path=cluster_path,nsf_hifigan_enhance=enhance)

    spk_list = list(spk_dict.keys())
    output_msg = "模型加载成功"
    return output_msg, gr.Dropdown.update(choices=spk_list, value=spk_choice)

def load_options():
    file_list = os.listdir("logs/44k")
    ckpt_list = []
    cluster_list = []
    for ck in file_list:
        if os.path.splitext(ck)[-1] == ".pth" and ck[0] != "k" and ck[:2] != "D_":
            ckpt_list.append(ck)
        if ck[0] == "k":
            cluster_list.append(ck)
    if not cluster_list:
        cluster_list = ["你没有聚类模型"]
    return choice_ckpt.update(choices = ckpt_list), config_choice.update(choices = os.listdir("configs")), cluster_choice.update(choices = cluster_list)

def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        print(input_audio)    
        sampling_rate, audio = input_audio
        print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        sf.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold)
        model.clear_empty()
        os.remove(temp_path)
        #构建保存文件的路径，并保存到results文件夹内
        timestamp = str(int(time.time()))
        output_file = os.path.join("results", sid + "_" + timestamp + ".wav")
        sf.write(output_file, _audio, model.target_sample, format="wav")
        return "Success", (model.target_sample, _audio)
    except Exception as e:
        if debug: traceback.print_exc()
        return "异常信息:"+str(e)+"\n请排障后重试",None

def vc_batch_fn(sid, input_audio_files, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold):
    global model
    try:
        if input_audio_files is None or len(input_audio_files) == 0:
            return "You need to upload at least one audio file", None
        if model is None:
            return "You need to upload a model", None
        for file_obj in input_audio_files:
            input_audio_path = file_obj.name
            #print(input_audio_path)
            audio, sampling_rate = sf.read(input_audio_path)
            #print(audio,audio.shape,sampling_rate)
            if np.issubdtype(audio.dtype, np.integer):
                audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
            elif not np.issubdtype(audio.dtype, np.floating):
                raise ValueError("Unsupported audio data type %r." % (audio.dtype,))
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.transpose(1, 0))
            _audio = model.slice_inference(input_audio_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling, enhancer_adaptive_key, cr_threshold)
            model.clear_empty()
            timestamp = str(int(time.time()))
            output_file_name = os.path.splitext(os.path.basename(input_audio_path))[0] + "_" + sid + "_" + timestamp + ".wav"
            output_file_path = os.path.join("results", output_file_name)
            sf.write(output_file_path, _audio, model.target_sample, format="wav")
        return "批量推理完成，音频已经被保存到results文件夹"
    except Exception as e:
        if debug: traceback.print_exc()
        return "异常信息:" + str(e) + "\n请排障后重试", None


def tts_func(_text,_rate,_voice):
    #使用edge-tts把文字转成音频
    # voice = "zh-CN-XiaoyiNeural"#女性，较高音
    # voice = "zh-CN-YunxiNeural"#男性
    voice = "zh-CN-YunxiNeural"#男性
    if ( _voice == "zh-CN-Xiaoyi" ) : voice = "zh-CN-XiaoyiNeural"
    if ( _voice == "zh-CN-Xiaoxiao" ) : voice = "zh-CN-XiaoxiaoNeural"
    if ( _voice == "zh-CN-Yunjian" ) : voice = "zh-CN-YunjianNeural"
    if ( _voice == "zh-CN-Yunxia" ) : voice = "zh-CN-YunxiaNeural"
    if ( _voice == "zh-CN-Yunyang" ) : voice = "zh-CN-YunyangNeural"
    if ( _voice == "zh-CN-liaoning-Xiaobei" ) : voice = "zh-CN-liaoning-XiaobeiNeural"
    if ( _voice == "zh-CN-shaanxi-Xiaoni" ) : voice = "zh-CN-shaanxi-XiaoniNeural"
    if ( _voice == "zh-HK-HiuGaai" ) : voice = "zh-HK-HiuGaaiNeural"
    if ( _voice == "zh-HK-HiuMaan" ) : voice = "zh-HK-HiuMaanNeural"
    if ( _voice == "zh-HK-WanLung" ) : voice = "zh-HK-WanLungNeural"
    if ( _voice == "zh-TW-HsiaoChen" ) : voice = "zh-TW-HsiaoChenNeural"
    if ( _voice == "zh-TW-HsiaoYu" ) : voice = "zh-TW-HsiaoYuNeural"
    if ( _voice == "zh-TW-YunJhe" ) : voice = "zh-TW-YunJheNeural"
    if ( _voice == "ja-JP-Keita" ) : voice = "ja-JP-KeitaNeural"
    if ( _voice == "ja-JP-Nanami" ) : voice = "ja-JP-NanamiNeural"
    if ( _voice == "en-GB-Libby" ) : voice = "en-GB-LibbyNeural"
    if ( _voice == "en-GB-Maisie" ) : voice = "en-GB-MaisieNeural"
    if ( _voice == "en-GB-Ryan" ) : voice = "en-GB-RyanNeural"
    if ( _voice == "en-GB-Sonia" ) : voice = "en-GB-SoniaNeural"
    if ( _voice == "en-GB-Thomas" ) : voice = "en-GB-ThomasNeural"
    if ( _voice == "en-US-Ana" ) : voice = "en-US-AnaNeural"
    if ( _voice == "en-US-Aria" ) : voice = "en-US-AriaNeural"
    if ( _voice == "en-US-Christopher" ) : voice = "en-US-ChristopherNeural"
    if ( _voice == "en-US-Eric" ) : voice = "en-US-EricNeural"
    if ( _voice == "en-US-Guy" ) : voice = "en-US-GuyNeural"
    if ( _voice == "en-US-Jenny" ) : voice = "en-US-JennyNeural"
    if ( _voice == "en-US-Michelle" ) : voice = "en-US-MichelleNeural"
    if ( _voice == "en-US-Roger" ) : voice = "en-US-RogerNeural"
    if ( _voice == "en-US-Steffan" ) : voice = "en-US-SteffanNeural"
    output_file = _text[0:10]+".wav"
    # communicate = edge_tts.Communicate(_text, voice)
    # await communicate.save(output_file)
    if _rate>=0:
        ratestr="+{:.0%}".format(_rate)
    elif _rate<0:
        ratestr="{:.0%}".format(_rate)#减号自带

    p=subprocess.Popen("edge-tts "+
                        " --text "+_text+
                        " --write-media "+output_file+
                        " --voice "+voice+
                        " --rate="+ratestr
                        ,shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE)
    p.wait()
    return output_file

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)#去除

def vc_tts_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,tts_voice,F0_mean_pooling,enhancer_adaptive_key,cr_threshold):
    #使用edge-tts把文字转成音频
    text2tts=text_clear(text2tts)
    output_file=tts_func(text2tts,tts_rate,tts_voice)

    #调整采样率
    sr2=44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    save_path2= text2tts[0:10]+"_44k"+".wav"
    wavfile.write(save_path2,sr2,
                (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                )

    #读取音频
    sample_rate, data=gr_pu.audio_from_file(save_path2)
    vc_input=(sample_rate, data)

    mes,b=vc_fn(sid, vc_input, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold)
    os.remove(output_file)
    os.remove(save_path2)
    return mes,b

def load_raw_dirs():
    #检查文件名
    allowed_pattern = re.compile(r'^[a-zA-Z0-9_@#$%^&()_+\-=\s]*$')
    for root, dirs, files in os.walk(raw_path):
        if root != raw_path:  # 只处理子文件夹内的文件
            for file in files:
                file_name, _ = os.path.splitext(file)
                if not allowed_pattern.match(file_name):
                    return "数据集文件名只能包含数字、字母、下划线"
    #检查有没有小可爱不用wav文件当数据集
    for root, dirs, files in os.walk(raw_path):
        if root != raw_path:  # 只处理子文件夹内的文件
            for file in files:
                if not file.endswith('.wav'):
                    return "数据集中包含非wav格式文件，请检查后再试"
    spk_dirs = []
    with os.scandir(raw_path) as entries:
        for entry in entries:
            if entry.is_dir():
                spk_dirs.append(entry.name)
    if len(spk_dirs) != 0:
        return raw_dirs_list.update(value=spk_dirs)
    else:
        return raw_dirs_list.update(value="未找到数据集，请检查dataset_raw文件夹")

def dataset_preprocess(branch_selection):
    if branch_selection == "v1":
        preprocess_commands = [
            r".\workenv\python.exe resample.py",
            r".\workenv\python.exe preprocess_flist_config.py",
            r".\workenv\python.exe preprocess_hubert_f0.py"
        ]
        # branch = "v1"
    if branch_selection == "vec768-layer12":
        preprocess_commands = [
            r".\workenv\python.exe resample.py",
            r".\workenv\python.exe preprocess_flist_config_768l12.py",
            r".\workenv\python.exe preprocess_hubert_f0_768l12.py"
        ]
        # branch = "vec768-layer12"        
    accumulated_output = ""

    for command in preprocess_commands:
        try:
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)

            accumulated_output += f"Command: {command}, Using Branch: {branch_selection}\n"
            yield accumulated_output, None, None

            for line in result.stdout:
                accumulated_output += line
                yield accumulated_output, None, None

            result.communicate()

        except subprocess.CalledProcessError as e:
            result = e.output
            accumulated_output += f"Error: {result}\n"
            yield accumulated_output, None, None

        accumulated_output += '-' * 50 + '\n'
        yield accumulated_output, None, None
    if branch_selection == "v1": 
        config_path = "configs/config.json"
    if branch_selection == "vec768-layer12":
        config_path = "configs/config_768l12.json" 
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    spk_dict = config["spk"]
    spk_name = config.get('spk', None)

    yield accumulated_output, gr.Textbox.update(value=spk_name), gr.Textbox.update(value=branch_selection, interactive=False)

def clear_output():
    return gr.Textbox.update(value="Cleared!>_<")

def config_fn(log_interval, eval_interval, keep_ckpts, batch_size, lr, fp16_run, all_in_mem, using_branch):
    config_origin = ""
    #print(using_branch)
    if using_branch == "":
        all_files = os.listdir("logs/44k")
        model_files = [f for f in all_files if f.startswith('G_') and f.endswith('.pth')]
        latest_model = max(model_files, key=lambda x: int(x[2:-4]))
        model_branch = get_model_info(latest_model)
        if model_branch == "v1":
            config_origin = ".\\configs\\config.json"
        elif model_branch == "Vec768-Layer12":
            config_origin = ".\\configs\\config_768l12.json"
        else:
            return "读取训练分支失败"
    elif using_branch == "v1":
        config_origin = ".\\configs\\config.json"
    elif using_branch == "vec768-layer12":
        config_origin = ".\\configs\\config_768l12.json"
    else:
        return "读取训练分支失败"
    #print(config_origin)
    with open(config_origin, 'r') as config_file:
        config_data = json.load(config_file)
    config_data['train']['log_interval'] = int(log_interval)
    config_data['train']['eval_interval'] = int(eval_interval)
    config_data['train']['keep_ckpts'] = int(keep_ckpts)
    config_data['train']['batch_size'] = int(batch_size)
    config_data['train']['learning_rate'] = float(lr)
    config_data['train']['fp16_run'] = fp16_run
    config_data['train']['all_in_mem'] = all_in_mem
    with open(config_origin, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)
    return "配置文件写入完成"

def training(gpu_selection, using_branch):
    if not os.listdir(r"dataset\44k"):
        return "数据集不存在，请检查dataset文件夹"
    dataset_path = "dataset/44k"
    no_npy_pt_files = True
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy') or file.endswith('.pt'):
                no_npy_pt_files = False
                break
    if no_npy_pt_files:
        return "数据集中未检测到f0和hubert文件，可能是预训练未完成"
    #备份logs/44k文件
    logs_44k = "logs/44k"
    pre_trained_model = "pre_trained_model"
    pre_trained_model_768l12 = "pre_trained_model/768l12"
    models_backup = "models_backup"
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    #new_backup_folder_number = next_backup_folder_number(models_backup)
    new_backup_folder = os.path.join(models_backup, str(timestamp))
    os.makedirs(new_backup_folder, exist_ok=True)
    for file in os.listdir(logs_44k):
        shutil.move(os.path.join(logs_44k, file), os.path.join(new_backup_folder, file))
    if using_branch == "v1":
        d_0_path = os.path.join(pre_trained_model, "D_0.pth")
        g_0_path = os.path.join(pre_trained_model, "G_0.pth")
    if using_branch == "vec768-layer12":
        d_0_path = os.path.join(pre_trained_model_768l12, "D_0.pth")
        g_0_path = os.path.join(pre_trained_model_768l12, "G_0.pth")
    if os.path.isfile(d_0_path) and os.path.isfile(g_0_path):
        print("D_0.pth and G_0.pth exist in pre_trained_model")
    else:
        print("D_0.pth and/or G_0.pth are missing in pre_trained_model")
    shutil.copy(d_0_path, os.path.join(logs_44k, "D_0.pth"))
    shutil.copy(g_0_path, os.path.join(logs_44k, "G_0.pth"))
    if using_branch == "v1":
        cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config.json -m 44k" % (gpu_selection)
    if using_branch == "vec768-layer12":
        cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config_768l12.json -m 44k" % (gpu_selection)
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"

def continue_training(gpu_selection):
    if not os.listdir(r"dataset\44k"):
        return "数据集不存在，请检查dataset文件夹"
    dataset_path = "dataset/44k"
    no_npy_pt_files = True
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy') or file.endswith('.pt'):
                no_npy_pt_files = False
                break
    if no_npy_pt_files:
        return "数据集中未检测到f0和hubert文件，可能是预训练未完成"
    all_files = os.listdir("logs/44k")
    model_files = [f for f in all_files if f.startswith('G_') and f.endswith('.pth')]
    if len(model_files) != 0:
        latest_model = max(model_files, key=lambda x: int(x[2:-4]))
        model_branch = get_model_info(latest_model)
    else: 
        return "你还没有已开始的训练"
    if model_branch == "Vec768-Layer12":
        cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config_768l12.json -m 44k" % (gpu_selection)
    elif model_branch == "v1":
        cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config.json -m 44k" % (gpu_selection)
    else:
        return "不受支持的模型"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"

def previous_selection_refresh():
    work_saved_list = []
    for entry in os.listdir("models_backup"):
        entry_path = os.path.join(models_backup_path, entry)
        if os.path.isdir(entry_path):
            work_saved_list.append(entry)
    return gr.Dropdown.update(choices=work_saved_list)


def kmeans_training():
    if not os.listdir(r"dataset\44k"):
        return "数据集不存在，请检查dataset文件夹"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", r".\workenv\python.exe cluster\train_cluster.py"])
    return "已经在新的终端窗口开始训练，训练聚类模型不会输出日志，检查任务管理器中python进程有在占用CPU就是正在训练，训练一般需要5-10分钟左右"

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles == None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"凸组合":0, "线性组合":1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"成功，文件被保存在了{path}"
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def updata_mix_info(files):
    try:
        if files == None : return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def pth_identify():
    if not os.path.exists(root_dir):
        return f"未找到{root_dir}文件夹，请先创建一个{root_dir}文件夹并按第一步流程操作"
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not model_dirs:
        return f"未在{root_dir}文件夹中找到模型文件夹，请确保每个模型和配置文件都被放置在单独的文件夹中"
    valid_model_dirs = []
    for path in model_dirs:
        pth_files = glob.glob(f"{root_dir}/{path}/*.pth")
        json_files = glob.glob(f"{root_dir}/{path}/*.json")
        if len(pth_files) != 1 or len(json_files) != 1:
            return f"错误: 在{root_dir}/{path}中找到了{len(pth_files)}个.pth文件和{len(json_files)}个.json文件。应当确保每个文件夹内有且只有一个.pth文件和.json文件"
        valid_model_dirs.append(path)
        
    return f"成功识别了{len(valid_model_dirs)}个模型：{valid_model_dirs}"

def onnx_export():
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    try:
        for path in model_dirs:
            pth_files = glob.glob(f"{root_dir}/{path}/*.pth")
            json_files = glob.glob(f"{root_dir}/{path}/*.json")
            model_file = pth_files[0]
            json_file = json_files[0]
            with open(json_file, 'r') as config_file:
                config_data = json.load(config_file)
            channels = config_data["model"]["gin_channels"]
            if str(channels) == "256":
                para1 = 1
            if str(channels) == "768":
                para1 = 192
            device = torch.device("cpu")
            hps = utils.get_hparams_from_file(json_file)
            SVCVITS = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model)
            _ = utils.load_checkpoint(model_file, SVCVITS, None)
            _ = SVCVITS.eval().to(device)
            for i in SVCVITS.parameters():
                i.requires_grad = False       
            n_frame = 10
            test_hidden_unit = torch.rand(para1, n_frame, channels)
            test_pitch = torch.rand(1, n_frame)
            test_mel2ph = torch.arange(0, n_frame, dtype=torch.int64)[None] # torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unsqueeze(0)
            test_uv = torch.ones(1, n_frame, dtype=torch.float32)
            test_noise = torch.randn(1, 192, n_frame)
            test_sid = torch.LongTensor([0])
            input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
            output_names = ["audio", ]
            onnx_file = os.path.splitext(model_file)[0] + ".onnx"
            torch.onnx.export(SVCVITS,
                              (
                                  test_hidden_unit.to(device),
                                  test_pitch.to(device),
                                  test_mel2ph.to(device),
                                  test_uv.to(device),
                                  test_noise.to(device),
                                  test_sid.to(device)
                              ),
                              onnx_file,
                              dynamic_axes={
                                  "c": [0, 1],
                                  "f0": [1],
                                  "mel2ph": [1],
                                  "uv": [1],
                                  "noise": [2],
                              },
                              do_constant_folding=False,
                              opset_version=16,
                              verbose=False,
                              input_names=input_names,
                              output_names=output_names)
        return "转换成功，模型被保存在了checkpoints下的对应目录"
    except Exception as e:
        if debug: traceback.print_exc()
        return "转换错误："+str(e)

# read ckpt list
file_list = os.listdir("logs/44k")
ckpt_list = []
cluster_list = []
for ck in file_list:
    if os.path.splitext(ck)[-1] == ".pth" and ck[0] != "k" and ck[:2] != "D_":
        ckpt_list.append(ck)
    if ck[0] == "k":
        cluster_list.append(ck)
if not cluster_list:
    cluster_list = ["你没有聚类模型"]

#read GPU info
ngpu=torch.cuda.device_count()
gpu_infos=[]
if(torch.cuda.is_available()==False or ngpu==0):if_gpu_ok=False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name=torch.cuda.get_device_name(i)
        if("MX"in gpu_name):continue
        if("10"in gpu_name or "16"in gpu_name or "20"in gpu_name or "30"in gpu_name or "40"in gpu_name or "A50"in gpu_name.upper() or "70"in gpu_name or "80"in gpu_name or "90"in gpu_name or "M4"in gpu_name or "T4"in gpu_name or "TITAN"in gpu_name.upper()):#A10#A100#V100#A40#P40#M40#K80
            if_gpu_ok=True#至少有一张能用的N卡
            gpu_infos.append("%s\t%s"%(i,gpu_name))
gpu_info="\n".join(gpu_infos)if if_gpu_ok==True and len(gpu_infos)>0 else "很遗憾您这没有能用的显卡来支持您训练"
gpus="-".join([i[0]for i in gpu_infos])

app = gr.Blocks()
with app:
    gr.Markdown(value="""
        ###sovits4.0 webui 推理&训练
                
        修改自原项目及bilibili@麦哲云和bilibili@羽毛布団

        仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容

        作者：bilibili@闲予1217

        """)
    with gr.Tabs():
        with gr.TabItem("推理"):
            with gr.Row():
                choice_ckpt = gr.Dropdown(label="模型选择", choices=ckpt_list, value="no_model")
                model_branch = gr.Textbox(label="模型分支", placeholder="请先选择模型", interactive=False)
            with gr.Row():
                config_choice = gr.Dropdown(label="配置文件", choices=os.listdir("configs"), value="no_config")
                config_info = gr.Textbox(label="输出信息", placeholder="请选择配置文件")
            cluster_choice = gr.Dropdown(label="选择聚类模型", choices=cluster_list, value="no_clu")
            enhance = gr.Checkbox(label="是否使用NSF_HIFIGAN增强,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭", value=False)
            refresh = gr.Button("刷新选项")
            loadckpt = gr.Button("加载模型", variant="primary")
            
            sid = gr.Dropdown(label="音色", value="speaker0")
            model_message = gr.Textbox(label="Output Message")
            
            choice_ckpt.change(get_model_info, [choice_ckpt], [model_branch])  
            config_choice.change(check_json, [config_choice, model_branch], [config_info])
            refresh.click(load_options,[],[choice_ckpt,config_choice,cluster_choice])
            loadckpt.click(load_model_func,[choice_ckpt,cluster_choice,config_choice,enhance,model_branch],[model_message, sid])

            gr.Markdown(value="""
                请稍等片刻，模型加载大约需要10秒。后续操作不需要重新加载模型
                """)
            with gr.Row():
                text2tts=gr.Textbox(label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪")
            with gr.Row():
                tts_voice = gr.Radio(label="声线/语种",choices=["zh-CN-Yunxi","zh-CN-Xiaoyi","zh-CN-Xiaoxiao","zh-CN-Yunjian","zh-CN-Yunxia","zh-CN-Yunyang","zh-CN-liaoning-Xiaobei","zh-CN-shaanxi-Xiaoni","zh-HK-HiuGaai","zh-HK-HiuMaan","zh-HK-WanLung","zh-TW-HsiaoChen","zh-TW-HsiaoYu","zh-TW-YunJhe","ja-JP-Keita","ja-JP-Nanami","en-GB-Libby","en-GB-Maisie","en-GB-Ryan","en-GB-Sonia","en-GB-Thomas","en-US-Ana","en-US-Aria","en-US-Christopher","en-US-Eric","en-US-Guy","en-US-Jenny","en-US-Michelle","en-US-Roger","en-US-Steffan"], value="zh-CN-Yunxi")
            with gr.Row():
                tts_rate = gr.Number(label="tts语速", value=0)
            with gr.Tabs():
                with gr.TabItem("单个音频上传"):
                    vc_input3 = gr.Audio(label="单个音频上传")
                with gr.TabItem("批量音频上传"):
                    vc_batch_files = gr.Files(label="批量音频上传", file_types=["audio"], file_count="multiple")
            with gr.Row():
                auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会跑调）", value=False)
                F0_mean_pooling = gr.Checkbox(label="F0均值滤波(池化)，开启后可能有效改善哑音，但有概率导致跑调", value=False)
                cr_threshold = gr.Number(label="F0过滤阈值，只有启动f0均值滤波时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音", value=0.05)
            with gr.Row():
                vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                cluster_ratio = gr.Number(label="聚类模型混合比例，0-1之间，默认为0不启用聚类，能提升音色相似度，但会导致咬字下降", value=0)
            with gr.Row():
                enhancer_adaptive_key = gr.Number(label="使NSF-HIFIGAN增强器适应更高的音域(单位为半音数)|默认为0", value=0,interactive=True)
                slice_db = gr.Number(label="切片阈值", value=-40)
            with gr.Accordion("高级设置（一般不需要动）", open=False):
                noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒/s", value=0)
                pad_seconds = gr.Number(label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
                lg_num = gr.Number(label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=0)
                lgr_num = gr.Number(label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75,interactive=True)
            
            with gr.Row():
                vc_submit = gr.Button("音频转换", variant="primary")
                vc_batch_submit = gr.Button("批量转换", variant="primary")
                tts_submit = gr.Button("文字转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")

        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])
        vc_batch_submit.click(vc_batch_fn, [sid, vc_batch_files, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold], [vc_output1])
        tts_submit.click(vc_tts_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,tts_voice,F0_mean_pooling,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])

        with gr.TabItem("训练"):
            gr.Markdown(value="""请将数据集文件夹放置在dataset_raw文件夹下，确认放置正确后点击下方获取数据集名称""")
            raw_dirs_list=gr.Textbox(label="Raw dataset directory(s):")
            get_raw_dirs=gr.Button("识别数据集", variant="primary")
            gr.Markdown(value="""确认数据集正确识别后请选择训练分支
                                 **v1**: 原版分支，so-vits-svc4.0的基础版本
                                 **vec768-layer12**: 特征输入更换为Content Vec的第12层Transformer输出，模型理论上会有更好的质量，但缺乏广泛验证
                                 注意，不同分支训练出来的模型不通用，并且对应不同的配置文件，在推理时选择不匹配的配置文件会导致错误
            """)
            with gr.Row():
                branch_selection = gr.Radio(label="选择训练分支", choices=["v1","vec768-layer12"], value="v1", interactive=True)
                raw_preprocess=gr.Button("数据预处理", variant="primary")
            preprocess_output=gr.Textbox(label="预处理输出信息，完成后请检查一下是否有报错信息，如无则可以进行下一步", max_lines=999)
            clear_preprocess_output=gr.Button("清空输出信息")
            with gr.Group():
                gr.Markdown(value="""填写训练设置和超参数""")
                with gr.Row():
                    gr.Textbox(label="当前使用显卡信息", value=gpu_info)
                    gpu_selection=gr.Textbox(label="多卡用户请指定希望训练使用的显卡ID（0,1,2...）", value=gpus, interactive=True)
                with gr.Row():
                    log_interval=gr.Textbox(label="每隔多少步(steps)生成一次评估日志", value="200")
                    eval_interval=gr.Textbox(label="每隔多少步(steps)验证并保存一次模型", value="800")
                    keep_ckpts=gr.Textbox(label="仅保留最新的X个模型，超出该数字的旧模型会被删除。设置为0则永不删除", value="10")
                with gr.Row():
                    batch_size=gr.Textbox(label="批量大小，每步取多少条数据进行训练，大batch可以加快训练但显著增加显存占用。6G显存建议设定为4", value="12")
                    lr=gr.Textbox(label="学习率，尽量与batch size成正比(6:0.0001)，无法整除的话四舍五入一下也行", value="0.0002")
                    fp16_run=gr.Checkbox(label="是否使用半精度训练，半精度训练可能降低显存占用和训练时间，但对模型质量的影响尚未查证", value=False)
                    all_in_mem=gr.Checkbox(label="是否加载所有数据集到内存中，硬盘IO过于低下、同时内存容量远大于数据集体积时可以启用", value=False)
                with gr.Row():
                    gr.Markdown("请检查右侧的说话人列表是否和你要训练的目标说话人一致，确认无误后点击写入配置文件，然后就可以开始训练了")
                    using_branch = gr.Textbox(label="当前使用训练分支", value="", interactive=False)
                    speakers=gr.Textbox(label="说话人列表")
                    write_config=gr.Button("写入配置文件", variant="primary")

            write_config_output=gr.Textbox(label="输出信息")

            gr.Markdown(value="""**点击从头开始训练**将会自动将已有的训练进度保存到models_backup文件夹，并自动装载预训练模型。
                **继续上一次的训练进度**将从上一个保存模型的进度继续训练。继续训练进度无需重新预处理和写入配置文件。
                """)
            with gr.Row():
                with gr.Column():
                    start_training=gr.Button("从头开始训练", variant="primary")
                    training_output=gr.Textbox(label="训练输出信息")
                with gr.Column():
                    continue_training_btn=gr.Button("继续上一次的训练进度", variant="primary")
                    continue_training_output=gr.Textbox(label="训练输出信息")
                with gr.Column():
                    kmeans_button=gr.Button("训练聚类模型", variant="primary")
                    kmeans_output=gr.Textbox(label="训练输出信息")

        with gr.TabItem("小工具/实验室特性"):
            gr.Markdown(value="""
                        ### So-vits-svc 4.0 小工具/实验室特性
                        提供了一些有趣或实用的小工具，可以自行探索
                        """)
            with gr.Tabs():
                with gr.TabItem("静态声线融合"):
                    gr.Markdown(value="""
                        <font size=2> 介绍:该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线 
                                          注意：
                                          1.该功能仅支持单说话人的模型
                                          2.如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个SpaekerID下的声音
                                          3.保证所有待混合模型的config.json中的model字段是相同的
                                          4.输出的混合模型可以使用待合成模型的任意一个config.json，但聚类模型将不能使用
                                          5.批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
                                          6.混合比例调整建议大小在0-100之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
                                          7.混合完毕后，文件将会保存在项目根目录中，文件名为output.pth
                                          8.凸组合模式会将混合比例执行Softmax使混合比例相加为1，而线性组合模式不会
                        </font>
                        """)
                    mix_model_path = gr.Files(label="选择需要混合模型文件")
                    mix_model_upload_button = gr.UploadButton("选择/追加需要混合模型文件", file_count="multiple", variant="primary")
                    mix_model_output1 = gr.Textbox(
                                            label="混合比例调整，单位/%",
                                            interactive = True
                                         )
                    mix_mode = gr.Radio(choices=["凸组合", "线性组合"], label="融合模式",value="凸组合",interactive = True)
                    mix_submit = gr.Button("声线融合启动", variant="primary")
                    mix_model_output2 = gr.Textbox(
                                            label="Output Message"
                                         )
                with gr.TabItem("onnx转换"):
                    gr.Markdown(value="""
                        提供了将.pth模型（批量）转换为.onnx模型的功能
                        源项目本身自带转换的功能，但不支持批量，操作也不够简单，这个工具可以支持在WebUI中以可视化的操作方式批量转换.onnx模型
                        有人可能会问，转.onnx模型有什么作用呢？相信我，如果你问出了这个问题，说明这个工具你应该用不上

                        ### Step 1: 
                        在整合包根目录下新建一个"checkpoints"文件夹，将pth模型和对应的json配置文件按目录分别放置到checkpoints文件夹下
                        看起来应该像这样：
                        checkpoints
                        ├───xxxx
                        │   ├───xxxx.pth
                        │   └───xxxx.json
                        ├───xxxx
                        │   ├───xxxx.pth
                        │   └───xxxx.json
                        └───……
                        """)
                    pth_dir_msg = gr.Textbox(label="识别待转换模型", placeholder="请将模型和配置文件按上述说明放置在正确位置")
                    #onnx_model_version = gr.Radio(label="选择训练分支", choices=["v1","Vec768-Layer12"], value="v1")
                    pth_dir_identify_btn = gr.Button("识别", variant="primary")
                    gr.Markdown(value="""
                        ### Step 2:
                        识别正确后点击下方开始转换，转换一个模型可能需要一分钟甚至更久
                        """)
                    pth2onnx_btn = gr.Button("开始转换", variant="primary")
                    pth2onnx_msg = gr.Textbox(label="输出信息")

                    mix_model_path.change(updata_mix_info,[mix_model_path],[mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button,mix_model_path], [mix_model_path,mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1,mix_mode], [mix_model_output2])
                    pth_dir_identify_btn.click(pth_identify, [], [pth_dir_msg])
                    pth2onnx_btn.click(onnx_export, [], [pth2onnx_msg])

        get_raw_dirs.click(load_raw_dirs,[],[raw_dirs_list])
        raw_preprocess.click(dataset_preprocess,[branch_selection],[preprocess_output, speakers, using_branch])
        clear_preprocess_output.click(clear_output,[],[preprocess_output])
        write_config.click(config_fn,[log_interval, eval_interval, keep_ckpts, batch_size, lr, fp16_run, all_in_mem, using_branch],[write_config_output])
        start_training.click(training,[gpu_selection, using_branch],[training_output])
        continue_training_btn.click(continue_training,[gpu_selection],[continue_training_output])
        kmeans_button.click(kmeans_training,[],[kmeans_output])

    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI设置</font>
                    """)
                debug_button = gr.Checkbox(label="Debug模式，反馈BUG需要打开，打开后控制台可以显示具体错误提示", value=debug)

        debug_button.change(debug_change,[],[])

        app.queue(concurrency_count=1022, max_size=2044).launch(server_name="127.0.0.1",inbrowser=True,quiet=True)