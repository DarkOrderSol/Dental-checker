Download the zip file from [releases](https://github.com/DarkOrderSol/Dental-checker/releases/tag/The_project)

This project contains the three trained models which are :-

1.Quadrant detection( identifies jaw quadrants ) 
2.Tooth detection( localizes individual teeth )  
3.Disease detection( disease detection )

, contains an validation folder that contains the testing subject for the models called (images) 

, contains a python script that detects the already trained models and deploys them using the validation folder as the test subject (app.py)

and to launch the app use the launch.bat, the app will launch  on the browser 
 
after launching this application , choose one of the validation images from the dropdown list and press run pipline 

an image will appear on the right with the detection of the models using borderline boxes with the diagnosis of the patient 

IMPORTANT BEFORE USAGE :-
 
install the following libraries :
gradio
ultralytics
pillow
numpy
torch
torchvision

here is a short code run in terminal to install them all :
pip install gradio ultralytics pillow numpy torch torchvision

and uncompress the file to use 


