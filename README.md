# BookBuddy
This is my capstone project at GMU. This BookBuddy application was developed to assist an aspiring author in promoting the content of her new book. The application offers three key functionalities: 
(1) AI-powered image and text generation, 
(2) web scraping to analyze search trends based on user-provided prompts.
(3) A business finance tracker that enables users to input and delete financial records while keeping track of monthly and total expenses.

The project was developed by a dedicated team including Alyssa Guzman, Alexander Roberts, Andrew Kubat, Ahmed Abdelghany, and myself. 

My contribution to the Capstone project: 
I was responsible for developing and executing the database functions of the project, specifically handling the MySQL-based components that support the finance management section. This involved designing efficient database schemas, writing and optimizing queries, and ensuring seamless data retrieval and storage. Additionally, I worked to maintain data integrity and performance, contributing to a robust and scalable financial management system. 

The below YouTube video presented by Alexander Roberts demonstrates the System Overview of the application. It describes how to use the functions of BookBuddy.

https://youtu.be/X9Erd0EyMoI?si=sAnukxGZBiT-cX89&t=300

The portion of the application I was responsible for in the video begins at 5:00 and provides an overview of the design and operational structure of the finance management system. 
![image](https://github.com/user-attachments/assets/c177e788-28ac-4145-b44b-2f603867dc98)

Please notes, running this application requires subscriptions to Google Search API (SerpApi) and Open API with having money in the OpenAI account. More information in the document below.



----------

Book Buddy Training Documentation

Section #	   Topic	                 Start Page

1	           Introduction	               	    2

2            	   System Setup	                    2

3            	   System Overview	            6

4	           Security                         7

5            	   Interface Design                 8

6            	   Troubleshooting and Help	    8



Section 1: Introduction

Welcome to the Book Buddy Training Documentation. This manual serves as your comprehensive guide, covering everything from the initial setup of your Book Buddy to troubleshooting the application. Whenever relevant, we will provide links and supplemental resources to enhance your understanding and implementation. For any queries regarding the software, please consult this document as your primary reference resource. Let's embark on this journey together to maximize your Book Buddy experience.

Section 2: System Setup

Local Resources and downloads needed prior to using Book Buddy.

1.	On the main project page for ‘Capstone,’ click on BookBuddy.py. To download the stable version of BookBuddy.py, click on development to open a drop down menu, then select master. Then, click the download icon(an arrow pointing down into a sideways bracket (This contains all of the system code).
2.	Click on Capstone to return to the main project page.
3.	Click on requirements.txt. Click on the download icon described in step 2 (this contains all of the modules you will need to install, instructions in a later section).

 Download Python 
*Skip this step if a Python 3 variation is already installed on your computer. Python 2
variations ARE NOT compatible with the program.

1.	Navigate to https://www.python.org/downloads/
2.	Click the yellow button “Download Python <latest version>” (at time of writing, that is 3.12.2).
2.1.	 
3.	Open Installer File that was saved to downloads.
4.	Go through setup, selecting the option “Install Now” unless customized install is necessary.
5.	With the exception of selecting  the “Add python.exe to path” option, remain on default options.
6.	When prompted to “Disable path length limit,” click on close to finish the installation.

Download PyCharm Community
Skip this step if Pycharm is already installed on your computer. Pycharm Professional also works (additional charge), however it is not needed for Book Buddy to operate. Refer to Section 5: Troubleshooting and Help for a video tutorial on setting up Pycharm

Pycharm is an essential IDE for Book Buddy to work. Book Buddy has been tested using Visual Studio code, which is not functional. Please always remember to run Book Buddy in Pycharm.

1.	Navigate to https://www.jetbrains.com/pycharm/download/other.html in a web browser.
2.	Download the PyCharm Community Edition variation “Windows (exe)” by clicking on the option matching it (at time of writing, “2023.3.5 - Windows (exe)”).
2.1.	 
3.	Open Installer File that was saved to downloads.
4.	Go through setup, keeping all default options.
5.	Exit the Installer.

	Setup SerpAPI
1.	Navigate to https://serpapi.com/
2.	Click Sign in or Register if you are a new user.
3.	Once you see your account dashboard, click “API Key” in the left vertical nav bar. The free  plan allows for 100 searches a month unless you upgrade your account which allows more. 
4.	Copy the API Key.
	Setup OpenAI API Key
1.	Navigate to https://openai.com/
2.	Either create an account, following the instructions on the website, or login with an existing account. 
3.	When logged in, there will be two choices, click the “API” option.
3.1.	 
4.	Click the lock icon on the left vertical nav menu.
5.	Click on the button that says “create new secret key.” 
6.	Follow the prompt.
7.	Before clicking create, be sure to write the password down somewhere safe and secure for reference. After the key is created, this is the only time it will be available to see and write down.
8.	Create the key.

Add money to OpenAI Account 
1.	Navigate https://openai.com/blog/openai-api
2.	Login to openAI account.
3.	Click on “Settings” in the left vertical nav bar.
4.	Click on “billing.”
5.	Click “Add credit balance.”
5.1.	 
6.	Add the desired amount to your account 
6.1.	Note: Adding 10 dollars will give up to 250 image generations or 5000 sets of text generation (up to 750 words each set). These values are the maximum if that is the only type being performed, mixing and matching the generation type will result in lower numbers for both.

	Setup SQL Server 
1.	Steps 2-6 can be skipped if the system already has the following installed:
1.1.	Visual Studio 2019 x64 redistributable
1.2.	Microsoft Visual C++ 14.0 or greater
2.	Navigate to https://visualstudio.microsoft.com/downloads/
3.	Download the ‘Community’ version of Visual Studio 2022 by clicking ‘Free download’.
4.	Run the Visual Studio Installer.
5.	In the Desktop & Mobile section, click the checkmark for ‘Desktop development with C++,’ then click install in the bottom right corner.
6.	One the install has finished, close the Visual Studio Installer.
7.	Navigate to https://dev.mysql.com/downloads/mysql/
8.	Download the ‘Windows (x86, 64-bit), MSI Installer’ by clicking the associated download button.
8.1.	 
9.	Follow the installer prompt, installing the ‘Typical setup’ option, unless otherwise necessary.
10.	Be sure to write the username and password down somewhere safe and secure for your reference. You will need this in a later section. 

	Setup Project 
1.	Open Pycharm 
2.	Create a new project 
3.	Name the project (ex: BookBuddyProject)
4.	Click and drag the BookBuddy.py file and Requirements.txt file from your downloads folder to the left hand project window in PyCharm.
5.	Open a new terminal 
6.	Run this command: pip install -r requirements. txt
7.	Run this command, replacing the red with your keys:
encryptoenv -a "OpenAIKey=InsertHere" "PASSWORD= InsertHere " "SerpAPI= InsertHere " -E
7.1.1.	OpenAIKey is the API key from OpenAI.
7.1.2.	PASSWORD is the SQL database password.
7.1.3.	SerpAPI is the API key from SerpAPI.
8.	In Pycharm, modify the Variables “HostNameVar” “userVar” and “Passwordvar” to the variables that match with what you wrote down. 
9.	Now you should be ready to run Book Buddy, if you have encountered any errors please refer to Section 5: Troubleshooting and Help
10.	Run the Code 
11.	If errors occur, ensure the previous steps were followed correctly. Otherwise, the Book Buddy Landing page should appear:
 
Section 3: System Overview

Video Training on the System overview and functionality.
	
1.	Open https://youtu.be/X9Erd0EyMoI for the system overview training video. 
1.1.	The video covers several topics including
1.1.1.	Interface Walkthrough of the Book Buddy functionalities:
1.1.1.1.	AI Content Generator 
1.1.1.2.	Web Scraper 
1.1.1.3.	Business Finance tracker 
1.1.1.4.	Running a Report 
1.1.2.	Button and link descriptions 
1.1.3.	Example usage of the functions 
1.1.4.	Destination file walkthrough
  	
Section 4: Security

General Password Security
a.	Requirements:
i.	14+ Characters long
ii.	1 of each of the following:
1.	Upper case letter
2.	Lower case letter
3.	Number
4.	Symbol
iii.	Cannot have 5+ occurrences of the same character
iv.	Changed annually
b.	How to change Windows account password:
i.	For a LOCAL account:
1.	Open the password menu in the settings application.
2.	Left click ‘Password’ to open a dropdown menu.
3.	Left click the ‘Change’ button.
4.	Enter the current password, then left click the ‘Next’ Button
5.	Enter the new password in the ‘New password’ field, and re-enter it in the ‘Confirm password’ field. Fill in the password hint, which cannot be left blank, then left click the ‘Next’ button.
6.	Left click the ‘Finish’ button to finalize the password reset.
ii.	For a MICROSOFT account:
1.	Open the password menu in the settings application.
2.	Left click ‘Password’ to open a dropdown menu.
3.	Left click the ‘Change’ button.
4.	Enter the current password, then click the ‘Next’ button.
5.	Enter the current password in the ‘Current Password’ field, then enter the new password in the ‘New Password’ field. Then click the ‘Next’ button.
c.	How to change MySQL password:
i.	Follow the instructions provided at: https://dev.mysql.com/doc/refman/8.0/en/resetting-permissions.html 
d.	Other third-party services:
i.	Log-in to the service in question, navigate to user account settings. Locate “change password”, and follow the instructions provided.
System Access
a.	Enable AutoSleep
i.	Open the ‘Power & Sleep Settings’ menu in the settings application.
ii.	Set both ‘When plugged in, turn off after’ and ‘When plugged in, PC goes to sleep after’ to the shortest time allowable. This decreases computer idle time before it locks.
b.	Lock computer whenever leaving (any of the following):
i.	Shortcut  (Windows key + L)
ii.	Right click the Windows icon, select ‘Shut down or Sign out’, then left-click the most relevant option (sleep to resume work soon, shut down if work is done for a while, and restart if necessary).

Section 5: Interface Design

Wireframes of the Book Buddy Interface design. This section serves as a reference of the Book Buddy Interface architecture. 

1.	Wireframes can be found here: https://www.figma.com/file/xndBfBaQ2EkMOfibfpiw2S/Book-Buddy-Interface-Design?type=design&node-id=0%3A1&mode=design&t=gnmZVvyk5w6OeOyA-1
1.1.
  	
Section 6: Troubleshooting and Help 

Errors that you may encounter, plus some external resources. 

Troubleshooting
Most errors encountered can be fixed by one of the following actions (not in order) 
-	Option 1: Delete your old env file from your project files in PyCharm.
-	Option 2: Ensure SQL Server is properly set up 
-	Option 3: Make sure you are using the most up-to-date version of Book Buddy and requirements text file
-	Option 4: Redownload Python 
-	Option 5: Ensure the encrypt command was configured correctly (refer to Section 2) 

Help 
1.	 Your IDE 
a.	Make sure you are only using Pycharm to run Book Buddy, IDE like Visual Studio Code will not work for this application 
2.	Operating System 
a.	Book Buddy only works on Windows operating system, this means the system will not work properly on MacOS. 
3.	“An exception Occured: List Index out of range”
a.	If you have redownloaded BookBuddy, be sure to delete your old env file from your project files in PyCharm. When you run the code, a new env file should appear and the error should be fixed 
4.	Business Finance Tracker not Running 
a.	Option 1: Delete your old env file from your project files in PyCharm. Re-run code
b.	Option 2: Ensure SQL Server is properly set up
5.	Can't find AI Generated content 
a.	Check the file path
6.	“Insufficient funds. Please add credits to your OpenAI Account 
a.	See Section about adding money to OpenAI account, this is needed to generate content. 
