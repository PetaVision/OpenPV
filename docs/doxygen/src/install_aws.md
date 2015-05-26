AWS Installation
==============================

Welcome to the cloud. Here's everything you need to know about how to get your favorite run on Amazon's AWS services.

[TOC]

Introduction
-------------------------------
Amazon AWS is a suite of tools to provide a virtual computing environment for end users. Here are the services we'll be using.

- EC2: Elastic Compute Cloud
   + Allows you to create a virtual instance that run on Amazon hardware.
- EBS: Elastic Block Store
   + The virtual hard drive that EC2 requires to do file I/O
- Snapshots
   + A snapshot of an EBS is essentially a copy of that hard drive.
- AMI: Amazon Machine Image
   + Snapshots can be converted to AMIs, which contain the operating system, tools, and in our case, the PetaVision toolbox.
- IAM: Identity and Access Management
   + Amazon's user account control that allows for user accounts for one AWS account.
- S3: Simple Storage Service
   + Long term shared storage. We don't have file io access to S3; everything must be downloaded/uploaded through https or AWS CLI. Databases and checkpoints will be stored here.

There exists a PetaVision AMI called 'PetaVision Public AMI' that has everything set up for PetaVision on GPUs. Additionally, the following software is installed on the image:
- Octave
- Python
- ffmpeg
- emacs

Overview
-------------------------------
Here's a quick overview of how our instance is set up.
- GPU spot instance
- 10GB root EBS that contains the OS, PetaVision and all software.
- 50 - 100GB personal EBS that will store sandboxes and output files.
- S3 for storing databases and checkpoints.

Services that cost money:
- GPU Spot Instance: market price, usually between $.10 and $.15 per hour
- EBS usage: $.10 per GB per Month
- S3 usage: $.03 per GB per Month
- Internet to EBS/S3 transfer: Free
- EBS/S3 transfer to Internet: cents per GB
- S3 to EBS and EBS to S3 transfers in the same region: Free

We are currently using the Oregon region.


First time initialization
-------------------------------
This section explains how to get a new user set up on the aws account to do runs.

###Activate your account ################
To create an AWS account, go to [Amazon's AWS page](http://aws.amazon.com) and click on Products.

If you are in Kenyon Lab, contact an account administrator to create and set up your account. You should get a user name and a temporary password starting up. Follow [these instructions](md_src_aws_pv_internal.html) for setting up a user in the PetaVision account.

###Create your ssh key pair ################
Each instance must have an ssh key for security. Here's how to generate your own.

Open a terminal and enter the following commands
~~~~~~~~~~~~~~~~~~~~~~~{.sh}
cd ~/.ssh #Create it if it doesn't exist
ssh-keygen -t rsa
~~~~~~~~~~~~~~~~~~~~~~~
Follow the instructions on the screen. Note the filename in which you saved the key, ex: username_aws.
~~~~~~~~~~~~~~~~~~~~~~~{.sh}
ssh-add ~/.ssh/username_aws
pbcopy < ~/.ssh/username_aws.pub #Copies contents of username_aws.pub to your clipboard for macs
~~~~~~~~~~~~~~~~~~~~~~~
Go to the EC2 management page, and find the option Key Pairs under Network and Security in the left tab.
Click Import Key Pair and paste the public key contents in. Click import.

Note that you may have to run ssh-add again if you get a public key error. To solve this, add this line to your `~/.bashrc` or `~/.profile`, replacing your private key name with username_aws
~~~~~~~~~~~~~~~~~~~~~~~{.sh}
ssh-add ~/.ssh/username_aws
~~~~~~~~~~~~~~~~~~~~~~~


Starting an Instance
-------------------------------
An AWS Instance allows the user to create a virtual environment to do runs. We do spot instances to save money, and already have the PetaVision capabilities to restart a run if it does die. For more information on spot instances, go to http://aws.amazon.com/ec2/purchasing-options/spot-instances/.

###Creating a Spot Instance  ##############
- Go to the EC2 management page. Click Instances under Instances in the left tab.
- Click Launch Instance.  Click on Community AMIs in the left tab.
- Using the search bar, find the AMI marked `PetaVision Public AMI`. Click Select.
- Pick a GPU instance (g2.2xlarge). Click next.
- (Note that there are additional steps involved if you are launching an mpi cluster of instances on this screen under the MPI Clusters section)
- Check Request Spot Instances under Purchasing option
- Type in an appropriate maximum price (approximately $0.15 - $0.20 above the current price is usually safe to insulate you from changes in price).
- Under subnet, pick one of the 3 choices based on the current price (Table when selecting a spot instance) and price history (Spot Requests under Instances, Price History up top. Select instance type as g2.2xlarge).
- Note the subnet.
- Click next.
- No additional storage is required. Make sure Delete on Termination is checked. Click next.
- No tags are needed. Click next.
- (Note that there are additional steps involved if you are launching an mpi cluster of instances on this screen under the MPI Clusters section)
- Check Select an existing security group. Select the default security group. Click Review and Launch.
- After reviewing, click launch.
- Select Choose an existing key pair, and select your key pair that you created earlier.
- Check the acknowledgement, and click Request Spot Instance.

Spot instances will take a few minutes to fulfill. Looking under Instances in the left tab, you can see the status of your request.
Once your instance is started, name the instance something relevant.

The PetaVision Public AMI already includes all the code and tools you will need to start a PetaVision. You will still wat


Working with a running instance
-------------------------------
- SSH into your running instance
   + Go to the EC2 management page. Click Instances under Instances in the left tab.
   + Find your instance in the list and note the Public IP address of the instance.
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
ssh ec2-user@public_ip_address
~~~~~~~~~~~~~~~~~~~~~~~~~

- After attached to your AWS run, update the instance:
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
sudo yum update
~~~~~~~~~~~~~~~~~~~~~~~~~

- Copy data to your instance
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
scp sourceFile ec2-user@public_ip_address:\~/destinationFile
~~~~~~~~~~~~~~~~~~~~~~~~~

Creating an EBS persistent volume
-------------------------------
AWS Spot Instances can die at any time. We will need an EBS persistent volume to store any run output we have as well as any sandboxes being ran from.

###Create your volume #############
- Go to the EC2 management page. On the left, click Volumes under Elastic Block Store. Click Create Volume.
- Adjust the size of the volume. 50 to 100 GB should be more than enough, depending on your run.
- Select the same subnet availability zone as your running instance.
- Click create.
- It will take a few minutes to create the volume. Name it something appropriate.

###Mounting your new volume to a running instance ##############
In order to use this volume in a running instance, we must mount it to a running instance.
- Go to the EC2 management page. On the left, click Volumes under Elastic Block Store.
- Select the volume you want to attach. The state of the volume should say available.
- Click actions, and select attach volume.
- Click on the Instance box and select your running instance.
- Make sure the device is set to `/dev/sdf`. This is the default value.

A new EBS volume is completely blank. That means we need to format the volume to our instance filesystem. Note that this needs to be done only once per new volume.
- Connect to your running instance.
~~~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
sudo mkfs -t ext4 /dev/sdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~


Setting up PetaVision
-------------------------------
Because the instance was booted from a PetaVision AMI, all of petavision should already exist in the home directory. There are several things needed to be done for PetaVision setup. 

###Accessing your mounted volume in a running instance ###############
- Connect to your running instance
- Run the file `~/startup.sh`
   + This will do 2 things: mount your data to the directory `~/mountData` and update PetaVision and PVSystemTests.

###Setting up your sandbox in your persistent volume ##############
- cd to mountData.
- Check out any sandboxes that you are planning on using. Ex:
   + svn co http://svn.code.sf.net/p/petavision/code/sandbox/HyPerHLCA HyPerHCLA            (this is read only)
   + svn co https://sourceForgeUserName@svn.code.sf.net/p/petavision/code/trunk PetaVision
   + svn co http://svn.code.sf.net/p/petavision/code/sandbox/PVSystemTests PVSystemTests    (this is a standard sandbox useful for checking PetaVision)
- cd to ~/workspace.
- Edit CMakeLists.txt using your favorite command-line text editor. (eg. vim or emacs)
- At the end of the file, add the absolute path to your sandbox to the cmakelists. Ex:
   + add_subdirectory(/home/ec2-user/mountData/HyPerHLCA /home/ec2-user/mountData/HyPerHLCA)
   + Note that the same directory is put twice into the command add_subdirectory.

###Building PetaVision ################
Note for those new to working with source code: the process of building a program takes the text/code description of a program and compiles that into a executable (0's and 1's) that a computer understands.  Think of it as creating a .exe file for a windows computer using a text description of what the program should do, which in our case is saved in the source files that end in .cpp and .hpp since we are working with C++. 

Compiling the program takes two steps. The first step involves creating the instructions for the compiler which will do the heavy lifting of making the program. We use the open-sourced software CMake to create those instructions using CMakeLists.txt as our input to CMake.  After using CMake, we will use a compiler (GCC on linux and Clang on the OSX) to actually make the executable program using the command 'make'.  The process is all quite easy, but hopefully this was useful information for you newbies. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
cd ~/workspace.
ccmake . 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of these variables should already be set up, but here are the important ones:
- CMAKE_BUILD_TYPE: Release
- CUDA_GPU: True
- CUDA_RELEASE: True
- CUDNN: True
- CUDNN_PATH: /home/ec2-user/cuDNN/cudnn-6.5-linux-x64-R2-rc1u
- Open_MP_THREADS: True

Press c to configure. Press e to exit the print statements.
Repeat until the g option appears and press g. 
cd to your sandbox and run `make -j 8`

S3
-------------------------------
S3 is long term storage for amazon's cloud. S3 objects (files, images) can only be accessed through http, but is much cheaper than ebs volumes.

###Create your access key ############
- Go to the IAM management page. On the left, click users.
- Select your account and click user actions.
- Select Manage Access Keys.
- Click create access key. Click Download Credentials and open the file. You will need the access key and secret access key.

###Create your bucket ##################
- Go to the S3 management page.
- See if your database is already there (such as kitti or neovision2heli datasets). If they are, you are done.
- On the top, click Create Bucket.
- Name your bucket something appropriate.
- Make sure your region is the same as your ec2 users. Transferring from region to region costs money.
- Click create.

###S3 Setup ##################
- SSH into your running instance.
- Enter the following command:
~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
aws configure
~~~~~~~~~~~~~~~~~~~~~~~~
- Enter your access key and secret access key. The rest can be left blank.


###S3 Transfer ################
There are several ways to upload data to S3 to an existing bucket.

####EBS Volume to S3 (preferred) #######################
- Getting your database on S3:
- Download your database, most likely to mountData. Make sure you're EBS volume is big enough to store all of the database.
- cd to the outer directory of your database.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
aws s3 cp --recursive myDatabaseFolder s3://yourBin/myDatabaseFolder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Note that you must put myDatabaseFolder on both the destination and source to keep the directory structure on s3.
- Optional: Make sure you have all of your database on s3. Run the same command as above except for sync instead of cp.
- Once the data is on s3, you can delete the local data off the EBS volume.

####Local to S3 ###########################
- Go to the S3 management page.
- Click an existing bucket that you would like your dataset to be, or create one and enter the bucket.
- Click upload.
- Drag the folder you would like to upload to the window.
- Click start upload.

Once you have your data on S3, you can specify a list of urls to feed to PetaVision Movie layer.
As an example:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   s3://myBin/myDatabaseFolder/img00.png
   s3://myBin/myDatabaseFolder/img01.png
   s3://myBin/myDatabaseFolder/img02.png
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


PetaVision Parameter Changes
-------------------------------
Here are the changes you need to make to your parameter file.
- In your movie layers, make sure your list of filenames is a list of s3 urls.
- Change your output directory to somewhere in mountData.
- Future work will be done to get checkpoints on s3.


MPI Clusters
-------------------------------
This section explains how to set up a cluster of instances to do a PetaVision run across multiple instances. Most of the instructions are the same, with several difference when launching instances:

###Configure Instance Details screen #########
- Set number of instances to the number of instances you would like to use in the cluster.
- Under Placement Group, create or select a placement group. All instance under one placement group have extra bandwidth between the nodes.
   + Note: If you chose an existing placement group, it must be in the same availability zone that you are requesting for the instance (e.g., "us-west-2a"). 

###Configure Security Group screen #############
When selecting an existing security group, use MPI_security instead of the default security group. This allows each instance to accept inbound data from other instances using this security group ID.

###Specific MPI cluster setup #############
- On your local machine, navigate to `petavision_trunk/docs/AWS/`
- Edit the file `hosts`
- Leave the first line alone. Add the external ip address for every node in your cluster. node0 will be your root node, the instance you will be launching everything from.
- Edit the file nodefile. Add each additional node you added to hosts, with the number of mpi processes you would like to run on as `slots`
- Run setup.sh in this directory on your local machine. This will move all the nessessary ssh keys to the clusters, as well as adding every node in the cluster into hosts.
   + This script will update and build PetaVision.
- ssh into your root node (node0).
- Follow the instructions to mount your EBS drive to your root node. Note that only the root node does any file io, so the root node is the only node that you need to attach and mount your volume to.
- Build your sandbox executable on the root node. Copy and move your sandbox executable somewhere *NOT* in your attached EBS volume (for example, in your home directory).
- Copy your executable to all nodes to your cluster. To do this, run the following from your home directory:
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
sh cpCluster.sh path_to_executable
~~~~~~~~~~~~~~~~~~~~~~~~~

To run from your root node, use the following command:
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
mpirun -np num_processors -hostfile ~/nodefile --bind-to none path_to_executable petavision_parameters
~~~~~~~~~~~~~~~~~~~~~~~~~
The `--bind-to none` flag is nessessary to get full threading utilization on aws.
