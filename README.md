# VideoBasedHandEye

A slicer module which preforms video based hand eye calibration of a system consisting of a webcam and optical tracker

Download/Setup Instruction:

- Create a copy of this repository on the local disk (C:) of your computer in the development file (D) or wherever else you see fit, and ensure all up to date versions of 3D Slicer and Plus Server Launcher are installed
- Launch slicer and download the OpenIGTLinkIF slicer module through slicer's extension manager, also make sure the sequences module is installed and useable
- To have this module show up as an option in the modules drop down, follow Edit> Application Settings> Modules and under additional module paths, click the >>  button on the right hand side
- This should show two buttons labeled "Add" and "Remove", click the add button and navigate to where the copy of this reposity is located on your computer, then click "Select Folder"
- The path to this module should then show up at the bottom of the Additional module paths list, and if so click the ok button at the bottom of the application settings window
- Restart slicer and check to soo if the module was properly added, you should find it under IGT in the drop down module list 
- To set up Plus for streaming in the data, open the Plus Server Launcher application and click the file button beside the Device set configuration directory textbox and navigate to the ConfigFiles folder within your copy of this repository and click "Select Folder"
- The device set drop down should automatically fill in with "PlusServer: Hand Eye Calibration", if not select it from the drop down options
- To launch this server, simply click the "Launch server" and wait for the "connection sucessful" message before proceeding



Usage Instructions:

- Open Slicer
- Lauch Hand Eye Calibration server through Plus server launcher application
- Open OpenIGTLinkIF slicer module under IGT, click the + under connectors tab, click Active checkbox, and ensure under I/O Configuration tab IGTLConnector shows ON in IGTL Type column
- Click triangle infront of IN and click the closed eye beside StylusTipToWebcam to show live 3D tracking of stylus, and if appropriate transform does not appear wave stylus with reflective fiducials visible in the optical tracker's field of view
- Hover over pin icon in top left of red slice window and choose Image_Reference (or its equivalent in your .xml file) in right drop down to show webcam feed
- If you have a previously obtained intrinsic matrix and set of distortion coefficients for the camera you are using, go to the HandEyeClaibration module under IGT and input those values in the appropriate user interface (UI) spaces, you can then skip recording the following checkerboard photos
- If the intrinsic matrix and distortion coefficients must still be obtained, open the Sequences slicer module and navigate to the "Undistortion" sequence browser, for the "Checkerboards" sequence and choose "Image_Referance" in the Proxy Node drop down, ensure the recording checkbox is checked
- If this or any other sequences do not automatically load, or get deleted, click the green plus button to create a new sequence and name it exactly as described
- Use the camera icon button to take at least 10 photos of a 8x6 or greater black and white checkerboard held in various positions around the photo frame (have at least one in each corner) and at various angles to be used to correct any camera distortions 
- Camera intrinsic matrix and undistortion constants will be automatically filled in to the appropriate spaces of the UI of the HandEyeCalibration module when you press the distortion calibration button
- In the "Data Collcetion" sequence browser,choose Image_Referance as the proxy node for the "Frames" sequence, and choose StylusTipToWebcam as the proxy node for the "Transforms" sequence, ensure the recording checkboxed are checker for both
- Ensure optical tracker has a clear view of the reflective fiducials on both the webcam and stylus, then click the red dot to record a video of the stylus tip being moved around within the image frame. Move the stylus slowly, try and cover as much of the camera's visual field as possible, and try to record at various depths away from the camera
- Back in the HandEyeCalibration module, click the Analyze button to return the extrinsic matrix, the average pixel, distance, and angular errors of all data points in that aquisition, as well as notifications for if tracking or circle detection was lost in any frames and which ones
