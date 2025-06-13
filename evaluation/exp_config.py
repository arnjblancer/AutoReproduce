#############################
ARXIV_ID = '2203.08679'
TASK='dataset:CIFAR100 teacher:resnet32x4 student:resnet8x4'
MODEL='dkd'
TITLE='Decoupled Knowledge Distillation'
METRIC='accuracy'
INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
The teacher model resnet32x4 checkpoint is under 'examples/dkd/source/resnet32x4_vanilla/ckpt_epoch_240.pth'
CIFAR100 training and testing dataset are under the folder 'examples/dkd/source/'.
There is no provided dataloader code. You need to build the training and testing dataloder by yourself.
Utilize {METRIC} as the evaluation metric.
"""

#############################
# ARXIV_ID = '2206.05099'
# TASK='MovingMnist'
# MODEL='simvp'
# TITLE='SimVP: Simpler yet Better Video Prediction'
# METRIC='MSE'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/simvp/source/train-images-idx3-ubyte.gz' and 'reproducebench/simvp/source/mnist_test_seq.npy' .
# Code related to process the data is provided to you and you could utilize the ```load_data()``` class to obtain the dataset and dataloader.
# Utilize {METRIC} as the evaluation metric.
# """

############################
# ARXIV_ID = '2301.12664'
# TASK='Darcy'
# MODEL='lsm'
# TITLE='Solving High-Dimensional PDEs with Latent Spectral Models'
# METRIC='MSE'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/lsm/source/piececonst_r421_N1024_smooth1.mat' and 'reproducebench/lsm/source/piececonst_r421_N1024_smooth12.mat' .
# Code related to process the data is provided to you and you could utilize the ```get_dataloader()``` class to obtain the dataset and dataloader.
# Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID='2310.06625'
# TASK='Traffic'
# MODEL='itransformer'
# TITLE='iTransformer: Inverted Transformers Are Effective for Time Series Forecasting'
# METRIC='MSE'
# INSTRUCTION = f"""
# You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper. 
# The experimental settings for both the prediction length and the fixed lookback length are both set to 96. The labeled length is set to 48.
# Data is under path 'reproducebench/itransformer/traffic.csv'.
# Code related to process the data is provided to you and you could utilize the ```data_provider()``` function to obtain the dataset and dataloader.
# Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID = '2309.14137'
# TASK='NYU-Depth-v2'
# MODEL='iebins'
# TITLE='IEBins: Iterative Elastic Bins for Monocular Depth Estimation'
# METRIC='threshold accuracy delta < 1.25$'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# The experimental settings for the pretrained backbone network is Swin-Tiny under the path 'reproducebench/iebins/source/swin_tiny_patch4_window7_224_22k.pth'. You should load pretrained model for experiment.
# Training and testing dataset are under the folder 'reproducebench/iebins/source/train/' and 'reproducebench/iebins/source/test/' with corresponding index file 'reproducebench/iebins/source/nyudepthv2_train_files_with_gt_dense.txt' and 'reproducebench/iebins/source/nyudepthv2_test_files_with_gt.txt'.
# Code related to process the data is provided to you and you could utilize the ```NewDataLoader()``` class to obtain the dataset and dataloader.
# Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID = '2111.08095'
# TASK='sine 20 precent'
# MODEL='timevae'
# TITLE='TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation'
# METRIC='predictor score'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training dataset are under the folder 'reproducebench/timevae/source/sine_subsampled_train_perc_20.npz'.
# Code related to process the data is provided to you and you could utilize the ```get_data()``` class to obtain the dataset.
# Utilize {METRIC} as the evaluation metric.
# """



#############################
# ARXIV_ID = '2108.11022'
# TASK='Citeseer full_supervised random_split'
# MODEL='tdgnn'
# TITLE='Tree Decomposed Graph Neural Network'
# METRIC='accuracy'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/tdgnn/source/Citeseer'.
# Code related to process the data is provided to you and you could utilize the ```get_dataset()``` class to obtain the dataset.
# Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID = '2105.05537'
# TASK='Synapse'
# MODEL='swinunet'
# TITLE='Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation'
# METRIC='mean dice'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/swinunet/source/Synapse'. 
# The pretrained backbone is swin_tiny_patch4_window7_224, you could access it from huggingface.
# Code related to process the data is provided to you and you could utilize the ```get_dataloader()``` class to obtain the dataset and dataloader.
# Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID = 'None'
# TASK='reside-indoor'
# MODEL='sfnet'
# TITLE='Selective Frequency Network for Image Restoration'
# METRIC='psnr'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/sfnet/source/reside-indoor'. 
# Code related to process the data is provided to you and you could utilize the ```train_dataloader()\nvalid_dataloader()\ntest_dataloader()\n```  class to obtain the dataset and dataloader.
# # Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID = '2302.03665'
# TASK='humaneva-I'
# MODEL='humanmac'
# TITLE='HumanMAC: Masked Motion Completion for Human Motion Prediction'
# METRIC='ADE'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/humanmac/source'. 
# Code related to process the data is provided to you and you could utilize the ```dataset_split()\nget_multimodal_gt_full()```  class to obtain the dataset and dataloader.
# # Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID = '2306.00306'
# TASK='LOLv1'
# MODEL='WBDM'
# TITLE='Low-Light Image Enhancement with Wavelet-based Diffusion Models'
# METRIC='ssim'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/WBDM/source'. 
# Code related to process the data is provided to you and you could utilize the ```AllWeatherDataset()```  class to obtain the dataset.
# # Utilize {METRIC} as the evaluation metric.
# """

#############################
# ARXIV_ID = '2211.09324'
# TASK='Gowalla'
# MODEL='bspm'
# TITLE='Blurring-Sharpening Process Models for Collaborative Filtering'
# METRIC='Recall'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/WBDM/source'. 
# Code related to process the data is provided to you and you could utilize the ```Loader()```  class to obtain the dataset.
# # Utilize {METRIC} as the evaluation metric.
# """


#############################
# ARXIV_ID = '2308.03364'
# TASK='Train:DF2K Test:Set5'
# MODEL='DAT-S'
# TITLE='Dual Aggregation Transformer for Image Super-Resolution'
# METRIC='PSNR'
# INSTRUCTION =f"""You are assigned an arXiv paper {TITLE} to replicate. You need to replicate the experiment conducted for {TASK} dataset in the paper.
# Training and testing dataset are under the folder 'reproducebench/DAT/source'. 
# Code related to process the data is provided to you and you could utilize the ```build_dataloader()```  class to obtain the dataset.
# # Utilize {METRIC} as the evaluation metric.
# """