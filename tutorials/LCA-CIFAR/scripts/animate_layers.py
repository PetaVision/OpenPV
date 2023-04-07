import numpy as np
import imageio
import pvtools

def animate_layers(filename: str, layer0: np.ndarray, layer1: np.ndarray=np.empty(0)):
  """animate_layers(filename, layer0, layer1)
  Creates an animated .gif file of one or two 4-D numpy arrays, in the
  numFrames-by-nx-by-ny-by-nf format returned by pvtools.readpvpfile.
  If the layer1 argument is present and nonempty, the two layers must have
  the same shape as each other, except possibly in the nx dimension.
  Each frame of the input data corresponds to one frame of the resulting .gif
  file.
  """
  numFrames = layer0.shape[0]
  nx = layer0.shape[1]
  ny = layer0.shape[2]
  nf = layer0.shape[3]

  if (layer1.size > 0):
    sizes_match = layer1.shape[0] == numFrames and layer1.shape[2] == ny and layer1.shape[3] == nf
    # error out if sizes_match is False

  gifFrames = []
  for frame in range(numFrames):
    frameData = layer0[frame]
    if (layer1.size > 0):
      frameData = np.vstack([frameData, layer1[frame]])
    minval = np.min(frameData)
    maxval = np.max(frameData)
    if np.isclose(minval, maxval):
      frameData[:] = 0.5
    else:
      frameData = (frameData - minval) / (maxval - minval)
    frameData_8bit = np.uint8(frameData * 255)
    gifFrames.append(frameData_8bit)

  imageio.mimsave(filename, gifFrames)
 

def command_line_help():
  """animate_layers layer0file [layer1file] outputfile  
  Loads one or two layer .pvp files and creates a frame-by-frame animated .gif
      layer0file [required] is the filename of the first .pvp file to read.
      layer1file [optional] is the name of the second layer .pvp files to read.
      outputfile [required] is the file name for the .gif output

      If the argument layer1file is present, the two layers are stacked
      vertically, and must have the same nx, nf, and numbers of frames.
  """


if __name__ == '__main__':
  import os
  import sys

  progpath = sys.argv[0]
  progname = os.path.basename(progpath)

  if len(sys.argv) <= 1:
    import pydoc
    print(pydoc.render_doc(command_line_help))
    exit(0)
  elif len(sys.argv) == 3:
    layer0 = pvtools.readpvpfile(sys.argv[1])['values']
    layer1 = np.empty(0)
    filename = sys.argv[2]
  elif len(sys.argv) == 4:
    layer0 = pvtools.readpvpfile(sys.argv[1])['values']
    layer1 = pvtools.readpvpfile(sys.argv[2])['values']
    filename = sys.argv[3]
  else:
    print(f'Error: {progname} requires either two or three arguments.', file=sys.stderr)
    import pydoc
    print(pydoc.render_doc(command_line_help), file=sys.stderr)
    exit(1)

  animate_layers(filename, layer0, layer1)
