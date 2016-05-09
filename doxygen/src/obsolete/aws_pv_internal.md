### Adding a user to the PetaVision AWS account ###

#### User instructions ####
To get added as a user in the PetaVision AWS account, you need to do the following:
* Contact a PetaVision administrator to get a username and temporary password.
* Go to <https://petavision.signin.aws.amazon.com/console>. Sign in using the username and temporary password.
* Amazon will prompt you to change your password.
* Return to <md_src_install_aws.html> to continue setting up the 

#### Administrator instructions ####
To add a user to the PetaVision AWS account, an administrator needs to do the following:

* Log into aws console.
* Under services, click IAM.
* On the left tab, click Users.
* Click Create New Users.
* Enter a new user name.
* Deselect Generate an access key for each user. Each user will create their own access keys to access s3. Click create.
* Find the user that was just created, select it, click actions, and select Manage Password.
* Select Assign an auto-generated password, and select Require user to create a new password at next sign-in. Click apply.
* Click Download Credentials and send the file to the new user.
* Select the user that was just created, select it, click actions, and select Add User to Group.
* Select Developer, and click Add to Groups.

