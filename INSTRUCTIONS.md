# Instructions

## Note on SSH

* If you have not done so already, you will need to create an [SSH key](http://docstore.mik.ua/orelly/networking_2ndEd/ssh/ch01_01.htm) for your linux account (e.g., in your SageCloud node):
    * In bash, enter the command ```ssh-keygen``` and hit Enter several times to generate the files ```~/.ssh/id_rsa``` and ```~/.ssh/id_rsa.pub```. These are your SSH keys.
    * Print the contents of your public key to the screen with ```cat ~/.ssh/id_rsa.pub```. Copy the contents (starting with ```ssh-rsa``` and ending with ```@somehostname```). Go to your GitHub account, click on the top-right corner menu, and go to Settings. In the "SSH and GPG keys" tab, add a "New SSH key", then paste your public key into the indicated "Key" text box, and give your new key a suitable "Title" so that you can recognize which computer it belongs to.
    * Finally, back in your original bash shell, enter ```ssh git@github.com``` to test that your key authenticates properly. If successful, you will see a message similar to ```Hi yourusername! You've successfully authenticated, but GitHub does not provide shell access.``` after which it will exit.
    * You should now be able to clone, pull, and push to GitHub directly from the terminal. (The light next to the SSH key in your GitHub account will also turn green.)

## Note on git Defaults
* To make your life easier, run the following commands in a bash terminal to set good default global options for ```git``` in your bash environment (this only has to be done once):
    * ```git config --global user.name "your full name here"```
    * ```git config --global user.email "youremail@here"```
    * ```git config --global push.default simple```
    * ```git config --global core.editor vim```

## This Assignment

* Clone this repository to your local machine.
* Edit the README.md file according to the instructions below.
* Perform a ```commit``` so your local repository remembers the edits you've made
* Now push the results back to the GitHub repository
* (See the tryGit resources in the info repository for more information.)

## General

* Edit the README.md file:
    * Change _\<your name\>_ and _\<date\>_ at the beginning of the file
    * Edit the line ```[![Build Status](https://travis-ci.com/chapman-phys220-2016f/UPDATETHIS.svg?token=CHANGEME&branch=master)](https://travis-ci.com/chapman-phys220-2016f/UPDATETHIS)``` replacing both instances of ```UPDATETHIS``` with the name of this repository (should be something like ```hw-1-<your git name>```), and the token with the correct security token taken from Travis-ci.com. (Note that for private repositories it is .com, while for public it is .org)
    * Add your own text in the __Assessment__ section. This text is taken seriously for continued course development, so be honest.
    * Reread the __Honor Pledge__ again and 'sign' it with your name at the end

* Connnect the repository to Travis CI
    * On the GitHub site, go to the upper-right menu, then to ```Integrations```
    * Click on ```Travis CI``` and Add the service. Authorize it if necessary.

* Correctly document each source file
    * Refer to the templates in the info repository for more information    

* Ensure that what you push to the github repository builds correctly (the build badge displays as [![Build Status](https://camo.githubusercontent.com/c71f5665277589f9ba8039c6e1b8bb120a3640b2/68747470733a2f2f696d672e736869656c64732e696f2f7472617669732f436861706d616e43505343323330537072696e6731362f41737369676e6d656e742d582e737667)]()) in the README file.
