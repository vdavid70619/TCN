#### Temporal Context Network for Activity Localization in Videos

This is the implementation for ICCV 17 paper "Temporal Context Network for Activity Localization in Videos".  
If you use the code, pretrained models, proposals, please cite:

@InProceedings{Dai_2017_ICCV,  
author = {Dai, Xiyang and Singh, Bharat and Zhang, Guyue and Davis, Larry S. and Qiu Chen, Yan},  
title = {Temporal Context Network for Activity Localization in Videos},  
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},  
month = {Oct},  
year = {2017}  
}  


#### Pre-trained Proposals

We provide the pre-trained proposal for both ActivityNet and THUMOS to assist future temporal detection works. 

Dataset|Link    
--- | ---  
ActivityNet | [Download](https://obj.umiacs.umd.edu/tcn_pretrained/actnet_proposal.7z)  
THUMOS | [Download](https://obj.umiacs.umd.edu/tcn_pretrained/thumos_proposal.7z)

#### Run the code 

Prerequisite: A caffe with python support  

Set PYTHONPATH to pycaffe path  
Set ACTNET_HOME to folder with features   

run "all_in_one.sh" to train and test

#### Pre-trained Features
We fine-tune [TSN](https://github.com/yjxiong/temporal-segment-networks) on dataset and extract score features.  

Dataset|Link    
--- | ---  
ActivityNet | [Download1](https://obj.umiacs.umd.edu/tcn_pretrained/actnet_tsn_score.7z.001) [Download2](https://obj.umiacs.umd.edu/tcn_pretrained/actnet_tsn_score.7z.002) [Download3](https://obj.umiacs.umd.edu/tcn_pretrained/actnet_tsn_score.7z.003)  
THUMOS | [Download](https://obj.umiacs.umd.edu/tcn_pretrained/thumos_tsn_score.7z)

For global features such as mbh and imagenet_shuffle, you can download from the official website.

#### Pre-trained Models
Here are the pre-trained models:

Dataset|Link    
--- | ---  
ActivityNet | [Download](https://obj.umiacs.umd.edu/tcn_pretrained/actnet_model.7z)  
THUMOS | [Download](https://obj.umiacs.umd.edu/tcn_pretrained/thumos_model.7z)
