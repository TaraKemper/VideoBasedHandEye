# VideoBasedHandEye

A slicer module which preforms video based hand eye calibration of a system consisting of a webcam and optical tracker

Download/Setup Instruction:

unfinished
- download OpenIGTLinkIF slicer module through slicer's extension manager


Usage Instructions:

- Open Slicer
- Lauch Hand Eye Calibration server through plus server launcher application
- Open OpenIGTLinkIF slicer module under IGT, click the + under connectors tab, click Active checkbox, and ensure under I/O Configuration tab IGTLConnector shows ON in IGTL Type column
- Click triangle infront of IN and click the closed eye beside StylusTipToWebcam to show live 3D tracking of stylus, and if appropriate transform does not appear wave stylus with reflective fiducials visible in the optical tracker's field of view
- Hover over pin icon in top left of red slice window and choose Image_Reference in right drop down to show webcam feed
- Open Sequences slicer module to begin recording video and tranform data to be analyzed
- Click large green + button to create new sequence, rename to "Checkerboards", and choose Image_Referance in the Proxy Node drop down
- Use the camera icon button to take at least 10 photos of a 8x6 or grater checkerboard held in various positions around the photo frame (have at least one in each corner) and at various angles to be used to correct any camera distortions
- Open an additional sequence browser by clicking on the Sequence Browser drop down menu and choosing "Create New Sequence Browser"
- In this sequence browser click the green + button twice, name the first new sequence "Frames" and choose Image_Referance as the proxy node, and name the second sequence "Transforms" and choose StylusTipToWebcam as the proxy node
- Ensure optical tracker has a clear view of the reflective fiducials on both the webcam and stylus, then click the red dot to record a video of the stylus tip being moved around within the image frame (move the stylus slowly and try and cover as much of the camera's visual field as possible) and the corresponding transforms
- Open the HandEyeCalibration slicer module under Examples, and click the Analyze button
