# 1.Requirements

## Environment

- Python 3.6.5
- PyTorch 0.4.1
- TorchVison 0.2.1
- FFmpeg, FFprobe

```bash conda create -n pt0.4.1 python=3.6.5```	#创建一个新环境  
```pip install --upgrade pip```	#更新pip  
```conda install numpy=1.15.4 pillow=5.3.0 scipy=1.1.0 scikit-learn=0.20.1 PyYAML=3.13 matplotlib=3.0.1 ffmpeg=4.0```	#安装所需要的包  
```pip install torch-0.4.1-cp36-cp36m-linux_x86_64.whl```	#安装pytorch  
```pip install torchvision==0.2.1```	#安装torchvision  
```pip install tensorboardX```	#安装tensorboardX  

# 2.Data Preparation
Basically, the processing of video data can be summarized into 3 steps:

- Get the videos and Decompress them into a folder
- Convert from videos to frames
- Construct file lists for training and validation

## Get the videos and Decompress them into a folder
###Kinetics
Because kinetics datasets are not original, so we first make label file in the same form as UCF-101:  
```
python video_txt.py <videodir_path> <txtfile_path>
```

###UCF-101
the ucf101 videos are archived in the downloaded file. Please use `unrar x UCF101.rar` to extract the videos.

###HMDB-51
the HMDB51 video archive has two-level of packaging. The following commands illustrate how to extract the videos.  
```mkdir rars && mkdir videos```  
```unrar x hmdb51_org.rar rars/```  
```for a in $(ls rars); do unrar x "rars/${a}" videos/; done;```

###Something-Somethingv1
the something-something-v1 frames are archived in the downloaded file. Please use `cat 20bn-something-something-v1-?? | tar zx` to decompress them into a folder.

###Something-Somethingv2
the something-something-v2 videos are archived in the downloaded file. Please use `cat 20bn-something-something-v2-?? | tar zx` to decompress them into a folder.
and you should use `python video_frame.py <videodir_path> <framedir_path>` to convert it from videos to frames.

###Jester
the jester-v1 frames are archived in the downloaded file. Please use `cat 20bn-jester-v1-?? | tar zx` to decompress them into a folder.

## Convert from videos to frames
```sh scripts/extract_frames.sh <videodir_path> <framedir_path>```  
or  
```python utils/video_frame.py <videodir_path> <framedir_path>```


## Construct file lists for training and validation
```
sh scripts/gen_dataset_list.sh <dataset_name>
```

# 3.Running the code

## Training

```
sh scripts/train_somethingv1.sh
```

## Testing

```
sh scripts/test_somethingv1.sh
```
