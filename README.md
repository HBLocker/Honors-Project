
Hons project deep learnring at detecting pgishing emails based form plain text CNN,BLSTM, BiCnn and a LSTM.

Would use the BiCNN as it was one of the quickest once trained. 

1:1 grade for dissertation. 

#TODO:
1 server to get email: this can be done via rust or python c (if needed) 
2 strips of non UTF-8 characters and special characters - python lib for this 
3 Passed to web filter
4 results come back for classficaiton 
5 sends back to user. 

Embedding file Location:
https://nlp.stanford.edu/projects/glove/


---> email ---> Deep Phish --> user 

Trying to make it a full stack kinda project now 

Runs within dokcer container 


User ---> "email" -->> Docker container  <----- Phish or non phish 
