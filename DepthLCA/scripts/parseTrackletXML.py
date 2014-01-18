#!/usr/bin/env python
"""
parse XML files containing tracklet info for kitti data base (raw data section)
(http://cvlibs.net/datasets/kitti/raw_data.php)

TODO:
* could add a few constants to give the truncation / occlusion / state values a meaning:
  if trackletObj.truncs[frameIdx] == TRUNCATED:
    ...
  elif trackletObj.occs[frameIdx,0] == NOT_OCCLUDED:
    ...
  elif trackletObj.states[frameIdx] == STATE_VALID:
    ...

No guarantees that this code is correct, usage is at your own risk!
"""

# Version History:
# 4/7/12 Christian Herdtweck: seems to work with a few random test xml tracklet files; 
#   converts file contents to ElementTree and then to list of Tracklet objects; 
#   Tracklet objects have str and iter functions

from sys import argv as cmdLineArgs
from xml.etree.ElementTree import ElementTree
import numpy as np
import itertools


class Tracklet(object):
  r""" representation an annotated object track 
  
  Tracklets are created in function parseXML and can most conveniently used as follows:

  for trackletObj in parseXML(trackletFile):
    for absoluteFrameNumber, translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders in trackletObj:
      ... your code here ...
    #end: for all frames
  #end: for all tracklets

  absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
  amtOcclusion and amtBorders could be None

  You can of course also directly access the fields objType (string), size (len-3 ndarray), firstFrame/nFrames (int), 
    trans/rots (nFrames x 3 float ndarrays), states/truncs (len-nFrames uint8 ndarrays), occs (nFrames x 2 uint8 ndarray),
    and for some tracklets amtOccs (nFrames x 2 float ndarray) and amtBorders (nFrames x 3 float ndarray). The last two
    can be None if the xml file did not include these fields in poses
  """

  objectType = None
  size = None  # len-3 float array: (height, width, length)
  firstFrame = None
  trans = None   # n x 3 float array (x,y,z)
  rots = None    # n x 3 float array (x,y,z)
  states = None  # len-n uint8 array of states
  occs = None    # n x 2 uint8 array  (occlusion, occlusion_kf)
  truncs = None  # len-n uint8 array of truncation
  amtOccs = None    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
  amtBorders = None    # None (n x 3) float array  (amt_border_l / _r / _kf)
  nFrames = None

  def __init__(self):
    r""" create Tracklet with no info set """
    self.size = np.nan*np.ones(3, dtype=float)

  def __str__(self):
    r""" return human-readable string representation of tracklet object

    called implicitly in 
    print trackletObj
    or in 
    text = str(trackletObj)
    """
    return '[Tracklet over {0} frames for {1}]'.format(self.nFrames, self.objectType)

  def __iter__(self):
    r""" returns an iterator that yields tuple of all the available data for each frame 
    
    called whenever code iterates over a tracklet object, e.g. in 
    for data in trackletObj:
      ...
    or
    trackDataIter = iter(trackletObj)
    """
    if self.amtOccs is None:
      return itertools.izip(self.trans, self.rots, self.states, self.occs, self.truncs, \
          itertools.repeat(None), itertools.repeat(None), xrange(self.firstFrame, self.firstFrame+self.nFrames))
    else:
      return itertools.izip(self.trans, self.rots, self.states, self.occs, self.truncs, \
          self.amtOccs, self.amtBorders, xrange(self.firstFrame, self.firstFrame+self.nFrames))
#end: class Tracklet


def parseXML(trackletFile):
  r""" parse tracklet xml file and convert results to list of Tracklet objects
  
  :param trackletFile: name of a tracklet xml file
  :returns: list of Tracklet objects read from xml file
  """

  # convert tracklet XML data to a tree structure
  eTree = ElementTree()
  print 'parsing tracklet file', trackletFile
  with open(trackletFile) as f:
    eTree.parse(f)

  # now convert output to list of Tracklet objects
  trackletsElem = eTree.find('tracklets')
  tracklets = []
  trackletIdx = -1
  nTracklets = None
  for trackletElem in trackletsElem:
    #print 'track:', trackletElem.tag
    if trackletElem.tag == 'count':
      nTracklets = int(trackletElem.text)
    elif trackletElem.tag == 'item_version':
      pass
    elif trackletElem.tag == 'item':
      trackletIdx += 1
      #print 'tracklet {0} of {1}'.format(trackletIdx, nTracklets)
      # a tracklet
      newTrack = Tracklet()
      isFinished = False
      hasAmt = False
      frameIdx = None
      for info in trackletElem:
        #print 'trackInfo:', info.tag
        if isFinished:
          raise ValueError('more info on element after finished!')
        if info.tag == 'objectType':
          newTrack.objectType = info.text
        elif info.tag == 'h':
          newTrack.size[0] = float(info.text)
        elif info.tag == 'w':
          newTrack.size[1] = float(info.text)
        elif info.tag == 'l':
          newTrack.size[2] = float(info.text)
        elif info.tag == 'first_frame':
          newTrack.firstFrame = int(info.text)
        elif info.tag == 'poses':
          # this info is the possibly long list of poses
          for pose in info:
            #print 'trackInfoPose:', pose.tag
            if pose.tag == 'count':   # this should come before the others
              if newTrack.nFrames is not None:
                raise ValueError('there are several pose lists for a single track!')
              elif frameIdx is not None:
                raise ValueError('?!')
              newTrack.nFrames = int(pose.text)
              newTrack.trans  = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              newTrack.rots   = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              newTrack.states = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
              newTrack.occs   = np.nan * np.ones((newTrack.nFrames, 2), dtype='uint8')
              newTrack.truncs = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
              newTrack.amtOccs = np.nan * np.ones((newTrack.nFrames, 2), dtype=float)
              newTrack.amtBorders = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              frameIdx = 0
            elif pose.tag == 'item_version':
              pass
            elif pose.tag == 'item':
              # pose in one frame
              if frameIdx is None:
                raise ValueError('pose item came before number of poses!')
              for poseInfo in pose:
                #print 'trackInfoPoseInfo:', poseInfo.tag
                if poseInfo.tag == 'tx':
                  newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                elif poseInfo.tag == 'ty':
                  newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                elif poseInfo.tag == 'tz':
                  newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                elif poseInfo.tag == 'rx':
                  newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                elif poseInfo.tag == 'ry':
                  newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                elif poseInfo.tag == 'rz':
                  newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                elif poseInfo.tag == 'state':
                  newTrack.states[frameIdx] = int(poseInfo.text)
                elif poseInfo.tag == 'occlusion':
                  newTrack.occs[frameIdx, 0] = int(poseInfo.text)
                elif poseInfo.tag == 'occlusion_kf':
                  newTrack.occs[frameIdx, 1] = int(poseInfo.text)
                elif poseInfo.tag == 'truncation':
                  newTrack.truncs[frameIdx] = int(poseInfo.text)
                elif poseInfo.tag == 'amt_occlusion':
                  newTrack.amtOccs[frameIdx,0] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_occlusion_kf':
                  newTrack.amtOccs[frameIdx,1] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_l':
                  newTrack.amtBorders[frameIdx,0] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_r':
                  newTrack.amtBorders[frameIdx,1] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_kf':
                  newTrack.amtBorders[frameIdx,2] = float(poseInfo.text)
                  hasAmt = True
                else:
                  raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
              frameIdx += 1
            else:
              raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
        elif info.tag == 'finished':
          isFinished = True
          if not hasAmt:
            newTrack.amtOccs = None
            newTrack.amtBorders = None
          tracklets.append(newTrack)
        else:
          raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))
    else:
      raise ValueError('unexpected tracklet info')
  #end: for tracklet list items

  return tracklets
#end: function parseXML


# when somebody runs this file as a script: print message
if __name__ == "__main__":
  parseXML(*cmdLineArgs[1:])

# (created using vim - the world's best text editor)
