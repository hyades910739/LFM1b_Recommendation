# CASER for music recommendation :
last update : Dec.12.2018  

This model is originally implemented by  [here](https://github.com/graytowne/caser_pytorch). To fit differnt type of data-format, I made some changes to the code, mostly in `music_interactions.py` and `train_caser.py`. 

Now containing :   

1. `caser.py` : main model implements by pytorch.  
2. `evaluation.py` : functions to evaluate model.  
3. `interaction.py` : class used to process data.  
4. `nusic_interactions.py` : extend from interaction.py, class `MusicInteraction` inherit `Interactions` to deal with my music data.  
5. `train_caser.py` : This is the main function, including training and testing process.  
6. `utils.py` : some help functions.   
