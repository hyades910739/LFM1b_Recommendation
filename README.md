# LFM-1b : music recommendation  
  

## Data : 
* LFM-1b Dataset : music listening events.  
source : [http://www.cp.jku.at/datasets/LFM-1b/](http://www.cp.jku.at/datasets/LFM-1b/)  
  
## Methods :  
1. caser : `caser_pytorch/`
  A CNN-based next-song prediction model.  
  Reference : Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, WSDM 2018.  
2. 

## Files in this folder:

1. `music_data_process.py` : process the LFM-LE data, split sequence to sub-sequences, and save to files.   
2. `music_data_examination.py` : check the overlaping rate : how many testing items doesn't appear in trainset.

  
