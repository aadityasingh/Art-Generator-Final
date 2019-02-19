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
4. To ssh into the instance, I recommend using the command line instead of their web client. You can get the ssh command from  the SSH drop down when you deploy your instance. It should look something like `gcloud compute --project "computingconnectionschallenge" ssh --zone "REGIONNAME" "INSTANCENAME-vm"`
5. Once in your instance, you'll need to make an NVIDIA NGC account and get an APIKEY and use that. To get the apikey, go to:https://ngc.nvidia.com/configuration/api-key after making an account.
6. Once you have this, go to your instance and do `docker login nvcr.io` and then `docker pull nvcr.io/nvidia/pytorch:18.02-py3`. This will take a few minutes as well.
7. Also on your VM, do `gcloud init` and `gcloud auth application-default login`. I'm not sure how necessary the second thing is, but it should help with getting the datafiles from the bucket. If you run into issues, you probably just need to do some commands of the form: `sudo chown -R $USER FOLDERTOCHANGE` to change the permissions to your user. Note, you might have to change permissions of the more outer folders first. 
8. At this point, you're ready to actually get our code and dataset. I recommend pulling the code from our github, then getting the data from the bucket. First do `git clone https://github.com/aadityasingh/Art-Generator-Final.git` from your home directory. Then, do `cd Art-Generator-Final` to get into the repo. Then run `gsutil -m cp -r gs://aaditya-cromagen-files/dfcvaegan dfcvaegan/`. For clarity, the -m flag allows gsutil to parallelize the port and the -r flag is for recursive copy. This command will take a bit (not more than 30 min). Now, do `cd  dfcvaegan` and `ls`. You should see a folder called data with the dataset. The reason things are organized this way (with a dfcvaegan folder, instead of just data) is so that we can
9. You're almost ready. Next step is compiling our custom docker image. For this, do `./rebuild.sh`.
10. After the image builds, you're ready to rumble, the command to run looks like `CUDA_VISIBLE_DEVICES=0 ./docker_run.sh python3 /dfcvaegan/main.py --run fiveClassFalseFalse --balance_classes 0 --random_crop 0 --movements Art_Nouveau_Modern Baroque Cubism Minimalism Naive_Art_Primitivism`. I'm going to dissect this command a bit for clarity. The CUDA_VISIBLE_DEVICES environment variable says which GPU (for a multi-GPU instance) to use. The docker_run script basically simplifies stuff like mounting volumes properly. Then, you just have your python3 command. Note that the filepath "/dfcvaegan/main.py" is a filepath INSIDE the docker container. This is where the file is in the volume mount. The command line arguments can be passed as normal.
11. I strongly recommend using tmux for runs. To start a new tmux session, just do `tmux` in command line. To start a session with a specific name, do `tmux new -s NAME`. In tmux sessions, you can basically run as normal. To exit a tmux session (and close it), just type `exit` in the session. To detach (aka get out of the session while stuff is running), do `ctrl+b` then click `d`. From your normal terminal, you can run `tmux ls` to see existing tmux sessions. To attach one of these, do `tmux a -t NAME`.
12. If you ever need to add more packages, this can be done by updating the Dockerfile in the docker/ folder. Afterwards, you should re-run `./rebuild.sh`. Also, you should commit these changes/any new packages you need for the master branch to run.

### GCP Post-run things:
1. As you must've noticed, after doing runs, all the runs get stored in a runs folder. To port this data back to your machine to view the images, first port your data to the bucket (so the whole team can see it). Run `gsutil -m cp -r runs/* gs://aaditya-cromagen-files/runs` from inside your dfcvaegan folder on your instance. Depending on  how many  runs you have to port, this can take a while! After this step, I recommend renaming the folder on your instance as old_runs, or just get rid of it (since it's all in the bucket now!). From here, you can use gsutil to copy onto your machine again.
2. As a reminder, to get things on your computer, just run `gsutil -m cp -r gs://aaditya-cromagen-files/runs/ runs/` from inside your local dfcvaegan folder.
3. Note that, although you could scp directly, this approach has two main advantages. First, your runs are available for vieiwng by the whole team on the bucket. Second, the gsutil -m option parallelizes the transfer, speeding up the process greatly.


## Deep Feature Consistent Variational Autoencoder (located in folder dfcvae/)
To run this, you will need to add a few folders. Create folders data/ runs/ samples/. In runs/, create folders checkpoints/ and logs/. Currently, the model is configured to run on three art categories. In the data/ folder, create a folder train3/. In that folder, create test/ and train/ folders. In each of these folders, add in 3 art style folders with the corresponding images. This is the folder we used for our Graduate Machine Learning project report. Since then, we have moved forward with the project, and the most relevant folder is DFCVAEGAN, the name originating from our use of the GAN discriminator as our deep feature extractor

## TODO VAEGAN folder

## TODO CNN folder

## Old files
To view some older versions/messier code that was the starting point of this project (before we cleaned it up), visit: https://github.com/eric-qian-d/Art-Generator/tree/master.








