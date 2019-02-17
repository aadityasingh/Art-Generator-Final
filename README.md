# Art-Generator-Final

We trained a variety of models. The one we use for generation is located in the dfcvae folder. A summary of this project/the project report is found in the pdf file.

Note that the models were all trained using the wikiart dataset, which is available here: TODO ADD LINK

## Dev folder (located in dfcvaegan/)
This is the folder where active project work is being done. This folder combines aspects of the other folders and basically aims to coalesce the codebases from different people. The only setup this folder 

For this part of the project, MIT graciously granted us some GCP credits to use. However, there is a lot more baggage to using GCP than there is in AWS, so I am documenting the setup below:

### GCP setup:
This guide is somewhat specific to the ComputingChallenge Project we were a part of. 
1. First, you  have to go find a region where you can make a GPU instance. You can use either Tesla V100s (which are slightly better, but not available everywhere) or Tesla P100s (more widely available I think). Go to the IAM page from the hamburger menu and go to Quotas. Under the metrics field, select the two types of GPUs to see availability. Then, pick a region where there is quota but no usage to use to launch your instance. DO NOT USE EUROPE-WEST4 THIS IS MY REGION AND I GOT A SPECIAL QUOTA INCREASE. If you want to make a machinge with multiptle GPUs, you should also request a quota increase.
2. Create your instance. I suggest using "nvidia gpu cloud image for deep learning and hpc" as opposed to the AISE instances because I couldn't get those up with extra memory (not sure what the issue was). The only modifications you need to make are: Set the region from part 1, Change memory to 64 GB (should be enough), and select the GPU type and quantity you want.
3. Install gcloud command client. You should be able to just run  `brew tap caskroom/cask`  and then `brew cask install google-cloud-sdk`, assuming you have homebrew. This step is optional if you choose to use google's web client; I just prefer commandline.
4. To ssh into the instance, I recommend using the command line instead of their web client. You can get the ssh command from  the SSH drop down when you deploy your instance. It should look something like `gcloud compute --project "computingconnectionschallenge" ssh --zone "REGIONNAME" "INSTANCENAME-vm`
5. Once in your instance, you'll need to make an NVIDIA NGC account and get an APIKEY and use that. To get the apikey, go to:https://ngc.nvidia.com/configuration/api-key after making an account.
6. Once you have this, go to your instance and do `docker login nvcr.io` and then `docker pull nvcr.io/nvidia/pytorch:18.02-py3`. This will take a few minutes as well.
7. Also on your VM, do `gcloud init` and `gcloud auth application-default login`. I'm not sure how necessary the second thing is, but it should help with getting the datafiles from the bucket. If you run into issues, you probably just need to do some commands of the form: `sudo chown -R $USER FOLDERTOCHANGE` to change the permissions to your user.
8. At this point, you're ready to actually get our code and dataset. This can be pulled from the bucket (the first time, after that I recommend just scp'ing the specific files you need to update). The command to do this is: `mkdir cromagen
gsutil -m cp -r cromagen gs://aaditya-cromagen-files`. For clarity, the -m flag allows gsutil to parallelize the port and the -r flag is for recursive copy. This command will take a bit (not more than 30 min). You should know have a dfcvaegan folder in your cromagen folder. The dfcvaegan folder contains all the files.
9. To actually run stuff, we're going to have some docker things that I haven't configured yet so hold off on running things. For the code changes you make right now, just running locally for a few iterations should help debug syntax errors/work for most things. I am working on docker right now


## Deep Feature Consistent Variational Autoencoder (located in folder dfcvae/)
To run this, you will need to add a few folders. Create folders data/ runs/ samples/. In runs/, create folders checkpoints/ and logs/. Currently, the model is configured to run on three art categories. In the data/ folder, create a folder train3/. In that folder, create test/ and train/ folders. In each of these folders, add in 3 art style folders with the corresponding images. This is the folder we used for our Graduate Machine Learning project report. Since then, we have moved forward with the project, and the most relevant folder is DFCVAEGAN, the name originating from our use of the GAN discriminator as our deep feature extractor

## TODO VAEGAN folder

## TODO CNN folder

## Old files
To view some older versions/messier code that was the starting point of this project (before we cleaned it up), visit: https://github.com/eric-qian-d/Art-Generator/tree/master.








