# ALTEGRAD 2019-2020

This project was done in the framework of the course *ALTERGRAD* taught by *M. Vazirigiannis*, during the first semester of the MVA master 2019-2020. Our team's name is *KILANI | NAOUMI*.

The public Leader board score of this provided solution is : **0.94551**

If you want to run everything from scratch (especially data building), you need the text/text folders in the data folder. (so as the script can access it and loop over the files )

Major bottlenecks in the provided script are : 

- The BERT features creation
- The Node2Vec features creation

Plus if you intend to run the .ipynb on Google Colab :

- You may face a problem when uploading the whole data (text/text being too big)
- Even if you manage to upload the entire files, Colab kind of disconnects and says that the files are inexisting even if they have been successfully uploaded.


To overcome these issues :

- BERT features : we ran them once on colab and saved them into .npy objects
- The Node2vec features : we ran the script once and saved them into a .csv
- Train/Test hosts : we ran them locally and saved them into .csv for further uses

You can find these files in the folder : data 

If those files are downloaded, read the comments in the .ipynb to see what are the lines that should be uncommented and ran and those that should be commented as there is no need to run them.

Actually the train_hosts_UPLOAD_DRIVE.csv is too big for github (i do not use GitLFS) so here is the drive link to download it :
https://drive.google.com/open?id=1l7sRDmgF0pd7oOMv_ZMBYLP--Ub_kSXB