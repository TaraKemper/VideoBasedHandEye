import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import cv2
import numpy as np
import math
import pathlib

#
# HandEyeCalibration
#

class HandEyeCalibration(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "HandEyeCalibration"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#HandEyeCalibration">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # HandEyeCalibration1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='HandEyeCalibration',
    sampleName='HandEyeCalibration1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'HandEyeCalibration1.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='HandEyeCalibration1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='HandEyeCalibration1'
  )

  # HandEyeCalibration2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='HandEyeCalibration',
    sampleName='HandEyeCalibration2',
    thumbnailFileName=os.path.join(iconsPath, 'HandEyeCalibration2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='HandEyeCalibration2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='HandEyeCalibration2'
  )

#
# HandEyeCalibrationWidget
#

class HandEyeCalibrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    
    self.saveFolder = pathlib.Path(__file__).parent.resolve()
    # Create all necessary folders
    os.makedirs(os.path.join(self.saveFolder, "OutputImages"), exist_ok=True)
    os.makedirs(os.path.join(self.saveFolder, "OutputImages", "Validation"), exist_ok=True)
    os.makedirs(os.path.join(self.saveFolder, "OutputImages", "Checkerboards"), exist_ok=True)
    os.makedirs(os.path.join(self.saveFolder, "OutputImages", "UndistortedCheckerboards"), exist_ok=True)
    os.makedirs(os.path.join(self.saveFolder, "OutputImages", "Undistortion"), exist_ok=True)
    os.makedirs(os.path.join(self.saveFolder, "OutputImages", "CircleDetection"), exist_ok=True)

  def UpdateTransforms(self, caller, event):
    print("TEST1")

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/HandEyeCalibration.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = HandEyeCalibrationLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # self.ui.pushButton.connect('clicked(bool)', self.testFunction)
    self.ui.pushButton.connect('clicked(bool)', self.AnalyzeVideo)
    self.ui.DistortionButton.connect('clicked(bool)', self.DistortionCalibration)
    self.ui.TemporalButton.connect('clicked(bool)', self.TemporalCalibration)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
    self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
    self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output volume"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
    self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
    self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
    self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:

      # Compute output
      self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
        self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

      # Compute inverted output (if needed)
      if self.ui.invertedOutputSelector.currentNode():
        # If additional output volume is selected then result with inverted threshold is written there
        self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
          self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()

  def testFunction(self):
    self.logic.logicTestFunction()

  def AnalyzeVideo(self):
    self.logic.logicAnalyzeVideo("Frames", "Transforms")

  def DistortionCalibration(self):
    self.logic.logicDistortionCalibration("Checkerboards")

  def TemporalCalibration(self):
    self.logic.logicTemporalCalibration()


#
# HandEyeCalibrationLogic
#

class HandEyeCalibrationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.saveFolder = pathlib.Path(__file__).parent.resolve()

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "100.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")

  def logicTestFunction(self):
    print ("Logic goes here")


  def logicDistortionCalibration(self, Checkerboards):
    """
    Logic function to preform a distortion calibration for the camera
    """
    print('got to distortion logic func')
    a = slicer.util.getNode(Checkerboards)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for count in range(a.GetNumberOfDataNodes()):
      # print('frame %d imported' % count)
      img = a.GetNthDataNode(count)
      img = slicer.util.arrayFromVolume(img)
      img = img[0,::-1,::-1,:]
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

      # Find the chess board corners
      ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

      # If found, add object points, image points (after refining them)
      if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7, 5), corners2, ret)
        _savePath = os.path.join(self.saveFolder, "OutputImages", "Checkerboards", "frame"+str(count)+".png")
        cv2.imwrite(_savePath, img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    for count in range(a.GetNumberOfDataNodes()):
      img = a.GetNthDataNode(count)
      img = slicer.util.arrayFromVolume(img)
      img = img[0, ::-1, ::-1, :]
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      # undistort
      h, w = img.shape[:2]
      newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
      img = cv2.undistort(img, mtx, dist, None, newcameramtx)

      # save raw undistorted image (check)
      _savePath = os.path.join(self.saveFolder, "OutputImages", "UndistortedCheckerboards", "frame"+str(count)+".png")
      cv2.imwrite(_savePath, img)

    return mtx, dist

  def logicTemporalCalibration(self):
    from scipy import signal

    CircleCentersDict, StylusTipCoordsDict = self.logicAnalyzeVideo("Frames2", "Transforms2")

    y1 = []
    y2 = []

    for i in range(len(CircleCentersDict)):
      y1.append(CircleCentersDict[i][1])
      y2.append(((StylusTipCoordsDict[i][0]*2.3)+385))

    print(y1)
    print(y2)

  def hand_eye_p2l(self, X, Q, A, tol=0.001):
    """
    INPUTS:   X : (3xn) 3D coordinates , ( tracker space) %
              Q : (2xn) 2D pixel locations (image space)
              A : (3x3) camera matrix
              tol : exit condition
    OUTPUTS:  R : 3x3 orthonormal rotation %
              t : 3x1 translation

    """
    import numpy as np
    import numpy.matlib

    n = Q.shape[1]
    e = np.ones(n)
    J = np.identity(n) - (np.divide((np.transpose(e) * e), n))
    Q = np.linalg.inv(A) @ np.vstack((Q, e))
    Y = ([[], [], []])
    for i in range(n):
      x = Q[:, i]
      y = np.linalg.norm(x)
      z = x / y
      z = np.reshape(z, (3, 1))
      Y = np.hstack((Y, z))
    Q = Y
    err = np.inf
    E_old = 1000 * np.ones((3, n))

    while err > tol:
      a = Y @ J @ X.T.conj()
      U, S, V = np.linalg.svd(a)
      R = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U @ V)]]) @ V #rotation
      T = Y - R @ X
      t = ([])
      for i in range(np.shape(Y)[0]):
        t = np.append(t, np.mean(T[i]))
      t = np.reshape(t, (np.shape(Y)[0], 1))
      h = R @ X + t * e
      H = ([])
      for i in range(np.shape(Q)[1]):
        H = np.append(H, np.dot(h[:, i], Q[:, i]))
      Y = np.matlib.repmat(H, 3, 1) * Q # reprojection
      E = Y - R @ X - t * e
      err = np.linalg.norm(E - E_old, 'fro')
      E_old = E

    return R, t

  def PixelValidation(self, T, X, Q, A):
    """
    INPUTS:  T : (4x4) hand eye calibration matrix
             X : (3xn) 3D coordinates , ( tracker space)
             Q : (2xn) 2D pixel locations (image space)
             A : (3x3) camera matrix
    OUTPUT:  pixelErrors : Column vector of pixel errors
    """
    pixels = []
    pixelErrors = []

    for k in range(X.shape[1]):
      point = X[:, k]
      point = np.reshape(point, (3, 1))

      pix = Q[:, k]
      pix = np.reshape(pix, (2, 1))

      point = np.vstack((point, 1))

      # Register 3D point to line
      cameraPoint = T @ point

      # Convert 3d point to homogeneous coordinates
      cameraPoint = cameraPoint / cameraPoint[2]
      cameraPoint = cameraPoint[0:2, :]
      cameraPoint = np.vstack((cameraPoint, 1))

      # Project point onto image using camera intrinsics
      pixel = A @ cameraPoint
      pixels.append(pixel)

      xError = abs(pixel[0, 0] - pix[0, 0])
      yError = abs(pixel[1, 0] - pix[1, 0])

      pixelErrors.append(np.sqrt(xError * xError + yError * yError))

    pixelErrors = np.reshape(pixelErrors, (X.shape[1], 1))
    return pixels, pixelErrors

  def DistanceValidation(self, T, X, Q, A):
    """
    INPUTS:   T : (4x4) hand eye calibration matrix
              X : (3xn) 3D coordinates , ( tracker space)
              Q : (2xn) 2D pixel locations (image space)
              A : (3x3) camera matrix
    OUTPUT:  distanceErrors : Column vector of distance errors
    """

    e = np.ones((1, X.shape[1]))
    Q = np.linalg.inv(A) @ np.vstack((Q, e))
    Y = ([[], [], []])
    for i in range(X.shape[1]):
      x = Q[:, i]
      y = np.linalg.norm(x)
      z = x / y
      z = np.reshape(z, (3, 1))
      Y = np.hstack((Y, z))
    Q = Y

    # Transform optical point to camera space
    X = T @ np.vstack((X, e))

    # Store the vector magnitude of each point
    mags = np.array([])
    for i in range(X.shape[1]):
      mags = np.append(mags, (np.sqrt(X[0, i] * X[0, i] + X[1, i] * X[1, i] + X[2, i] * X[2, i])))

    mags = np.reshape(mags, (X.shape[1], 1))

    # Normalize vector
    Y = ([[], [], [], []])
    for i in range(X.shape[1]):
      x = X[:, i]
      y = np.linalg.norm(x)
      z = x / y
      z = np.reshape(z, (4, 1))
      Y = np.hstack((Y, z))
    X = Y

    distanceErrors = []

    for k in range(X.shape[1]):
      x = X[0:3, k]
      q = Q[:, k]

      rot_axis = np.cross(x, q) / np.linalg.norm(np.cross(x, q))
      rot_angle = math.acos(np.dot(x, q) / (np.linalg.norm(x) * np.linalg.norm(q)))
      R = np.hstack((rot_axis, rot_angle))

      angle = rot_angle

      distanceErrors.append(mags[k, 0] * np.tan(angle))

    distanceErrors = np.reshape(distanceErrors, ((X.shape[1], 1)))
    return distanceErrors

  def AngularValidation(self, T, X, Q, A):
    """
    Hand Eye calibration angular validation
    INPUTS:   T : (4x4) hand eye calibration matrix
              X : (3xn) 3D coordinates , ( tracker space)
              Q : (2xn) 2D pixel locations (image space)
              A : (3x3) camera matrix
    OUTPUT:  angularErrors : Column vector of angular errors
    """
    e = np.ones((1, X.shape[1]))
    Q = np.linalg.inv(A) @ np.vstack((Q, e))
    Y = ([[], [], []])
    for i in range(X.shape[1]):
      x = Q[:, i]
      y = np.linalg.norm(x)
      z = x / y
      z = np.reshape(z, (3, 1))
      Y = np.hstack((Y, z))
    Q = Y

    # Transform optical point to camera space
    X = T @ np.vstack((X, e))

    # Normalize vector
    Y = ([[], [], [], []])
    for i in range(X.shape[1]):
      x = X[:, i]
      y = np.linalg.norm(x)
      z = x / y
      z = np.reshape(z, (4, 1))
      Y = np.hstack((Y, z))
    X = Y

    angularErrors = []

    for k in range(X.shape[1]):
      x = X[0:3, k]
      q = Q[:, k]

      rot_axis = np.cross(x, q) / np.linalg.norm(np.cross(x, q))
      rot_angle = math.acos(np.dot(x, q) / (np.linalg.norm(x) * np.linalg.norm(q)))
      R = np.hstack((rot_axis, rot_angle))

      angle = rot_angle

      angularErrors.append(math.degrees(angle))

    angularErrors = np.reshape(angularErrors, ((X.shape[1], 1)))
    return angularErrors

  def logicAnalyzeVideo(self, Frames, Transforms):
    """
    Read through video frame by frame and analyze each using colour thresholding and hough transform
    same as original function except applying a binary mask after colour thresholding to improve accuracy of circle detection
    """
    a = slicer.util.getNode(Frames)

    markupsNode = None
    try:
      markupsNode = slicer.util.getNode("HECFiducials")
      # Clear the list
      markupsNode.RemoveAllControlPoints()
    except:
      markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
      markupsNode.SetName("HECFiducials")

    StylusTipCoordsX = ([])
    StylusTipCoordsY = ([])
    StylusTipCoordsZ = ([])

    CircleCentersX = ([])
    CircleCentersY = ([])

    #mtx, dist = self.logicDistortionCalibration("Checkerboards")
    #mtx = np.array([[622.97040331 0. 322.60669888], [0. 619.81629638 239.44681572], [ 0. 0. 1. ])
    mtx = np.array([[622.97040331, 0, 322.60669888], [0, 619.81629638, 239.44681572], [0, 0, 1]]);
    dist = np.array([1.10517578e-01, -2.65829231e-01, -1.90690252e-05, 7.00236680e-05, 1.39000650e-01])

    for count in range(a.GetNumberOfDataNodes()):
      img = a.GetNthDataNode(count)
      img = slicer.util.arrayFromVolume(img)
      img = img[0,::-1,::-1,:]
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      # undistort
      h, w = img.shape[:2]
      newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
      img = cv2.undistort(img, mtx, dist, None, newcameramtx)

      ## save raw undistorted image (check)
      #_savePath = os.path.join(self.saveFolder, "OutputImages", "Undistortion", "frame"+str(count)+".png")
      #cv2.imwrite(_savePath, img)

      # Colour Threshold for green
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(hsv, (30, 50, 0), (80, 255, 255))
      target = cv2.bitwise_and(img, img, mask=mask)

      # Apply binary mask
      gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
      th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

      # Find Circle
      blurred = cv2.medianBlur(binary, 25)
      blurred = cv2.blur(blurred, (10, 10))
      circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 0.1, 1000, param1=50, param2=30, minRadius=0, maxRadius=1000)

      # save thresholded image (binary or blurred) which the hough transform is being applied to (check)
      _savePath = os.path.join(self.saveFolder, "OutputImages", "Undistortion", "frame"+str(count)+".png")
      cv2.imwrite(_savePath, blurred)
        
      c = vtk.vtkMatrix4x4()
      slicer.util.getNode(Transforms).GetNthDataNode(count).GetMatrixTransformToWorld(c)
      x = c.GetElement(0, 3)
      y = c.GetElement(1, 3)
      z = c.GetElement(2, 3)

      # draw fiducials following spatial tracking for visual validation
      markupsNode.AddFiducial(x,y,z)

      # Draw Calculated Circle
      if circles is None:
        print('No circles detected in frame %d' % count)
      else:
        # Convert the circle parameters a, b and r to integers.
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
          # draw the outer circle
          cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
          # draw the center of the circle
          cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
          center = [i[0], i[1]]

        #check that no 3d coords from the optical tracker are repeated (if tracking is lost) and append appropriate values to lists
        
        #c = vtk.vtkMatrix4x4()
        #slicer.util.getNode(Transforms).GetNthDataNode(count).GetMatrixTransformToWorld(c)
        #x = c.GetElement(0, 3)
        #y = c.GetElement(1, 3)
        #z = c.GetElement(2, 3)

        ## draw fiducials following spatial tracking for visual validation
        #markupsNode = slicer.util.getNode("MarkupsFiducial")
        #markupsNode.AddFiducial(x,y,z)

        #if else statement to deal with when no elements have yet been appended and thus checking if the last element of the list matches the one being appended results in an error
        if len(StylusTipCoordsX)>0:
          if StylusTipCoordsX[len(StylusTipCoordsX)-1] == x:
            print("Spatial tracking lost in frame %d" % count)
          else:
            #add circle centers to list
            CircleCentersX = np.append(CircleCentersX, center[0])
            CircleCentersY = np.append(CircleCentersY, center[1])
            #add corresponding transforms to list
            slicer.util.getNode(Transforms).GetNthDataNode(count).GetMatrixTransformToWorld(c)
            StylusTipCoordsX = np.append(StylusTipCoordsX, x)
            StylusTipCoordsY = np.append(StylusTipCoordsY, y)
            StylusTipCoordsZ = np.append(StylusTipCoordsZ, z)

            # Write new image to output file with circles drawn
            _savePath = os.path.join(self.saveFolder, "OutputImages", "CircleDetection", "frame"+str(count)+".png")
            cv2.imwrite(_savePath, img)
        else:
          # add circle centers to list
          CircleCentersX = np.append(CircleCentersX, center[0])
          CircleCentersY = np.append(CircleCentersY, center[1])
          # add corresponding transforms to list
          slicer.util.getNode(Transforms).GetNthDataNode(count).GetMatrixTransformToWorld(c)
          StylusTipCoordsX = np.append(StylusTipCoordsX, x)
          StylusTipCoordsY = np.append(StylusTipCoordsY, y)
          StylusTipCoordsZ = np.append(StylusTipCoordsZ, z)

          # Write new image to output file with circles drawn
          _savePath = os.path.join(self.saveFolder, "OutputImages", "CircleDetection", "frame"+str(count)+".png")
          cv2.imwrite(_savePath, img)

    #Format input matricies and output rotation and translational calibration matrix
    StylusTipCoords = np.vstack(( StylusTipCoordsX, StylusTipCoordsY, StylusTipCoordsZ))
    _savePath = os.path.join(self.saveFolder, "OutputImages", "StylusTipCoordsOutput.txt")
    np.savetxt(_savePath, StylusTipCoords, delimiter =", ", newline = "\n \n")
    # print (StylusTipCoords)

    CircleCenters = np.vstack(( CircleCentersX, CircleCentersY))
    _savePath = os.path.join(self.saveFolder, "OutputImages", "CircleCentersOutput.txt")
    np.savetxt(_savePath, CircleCenters, delimiter =", ", newline = "\n \n")
    # print(CircleCenters)

    # print(newcameramtx)

    R,t = self.hand_eye_p2l(StylusTipCoords, CircleCenters, newcameramtx)
    calibration = np.vstack((np.hstack((R,t)),[0,0,0,1]))
    # print(calibration)
    

    print("")

    pixels,pixelErrors = self.PixelValidation(calibration, StylusTipCoords, CircleCenters, newcameramtx)
    print("Average pixel error:", "%.2f pixels" % (sum(pixelErrors)/pixelErrors.shape[0])[0])

    distanceErrors = self.DistanceValidation(calibration, StylusTipCoords, CircleCenters, newcameramtx)
    print("Average distance error:", "%.2f mm" % (sum(distanceErrors) / distanceErrors.shape[0])[0])

    angularErrors = self.AngularValidation(calibration, StylusTipCoords, CircleCenters, newcameramtx)
    print("Average angular error:", "%.2f degrees" % (sum(angularErrors) / angularErrors.shape[0])[0])

    for i in range(len(pixels)):
      try:
        _filePath = os.path.join(self.saveFolder, "OutputImages", "CircleDetection", "frame"+str(i)+".png")
        img = cv2.imread(_filePath)
        if pixels[i][0] != -1:
          cv2.circle(img, (pixels[i][0], pixels[i][1]), 1, (0, 255, 255), 2)
          _filePath = os.path.join(self.saveFolder, "OutputImages", "Validation", "frame"+str(i)+".png")
          cv2.imwrite(_filePath, img)
      except:
        pixels.insert(i-1, np.array([[-1],[-1],[-1]]))







  def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

#
# HandEyeCalibrationTest
#

class HandEyeCalibrationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_HandEyeCalibration1()

  def test_HandEyeCalibration1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('HandEyeCalibration1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = HandEyeCalibrationLogic()

    # Test algorithm with non-inverted threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.process(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')