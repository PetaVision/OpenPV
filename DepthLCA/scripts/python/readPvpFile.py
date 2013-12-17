import numpy as np
import struct

def readHeaderFile(filestream, pos=None):
   params = struct.unpack("iiiiiiiiiiiiiiiiiid", filestream.read(80))
   header = {}
   header["headersize"] = params[0]
   header["numparams"]  = params[1]
   header["filetype"]   = params[2]
   header["nx"]         = params[3]
   header["ny"]         = params[4]
   header["nf"]         = params[5]
   header["numrecords"] = params[6]
   header["recordsize"] = params[7]
   header["datasize"]   = params[8]
   header["datatype"]   = params[9]
   header["nxprocs"]    = params[10]
   header["nyprocs"]    = params[11]
   header["nxGlobal"]   = params[12]
   header["nyGlobal"]   = params[13]
   header["kx0"]        = params[14]
   header["ky0"]        = params[15]
   header["nb"]         = params[16]
   header["nbands"]     = params[17]

   if header["numparams"] > 20:
      header["additional"] = []
      numAddParams = header["numparams"] - 20
      for i in range(numAddParams):
         header["additional"].append(struct.unpack("i", filestream.read(4)))

   assert header["datatype"] == 3
   assert (header["filetype"] == 4 or header["filetype"] == 3)
   return header

#Returns a matrix
def readData(filestream, shape, numPerFrame):
   #Read timestep
   timestamp = filestream.read(8)
   if len(timestamp) != 8:
      #EOF
      return (-1, None)
   idx = struct.unpack("d", timestamp)
   outmat = np.fromfile(filestream, np.float32, numPerFrame).reshape(shape)
   return (idx, outmat)


#Testing
if __name__ == "__main__":
   filename = "/nh/compneuro/Data/Depth/depth_data_1/pvp/depth_00.pvp"

   f = open(filename, 'rb')
   header = readHeaderFile(f)
   shape = (header["ny"], header["nx"], header["nf"])
   numPerFrame = shape[0] * shape[1] * shape[2]
   data = readData(f, shape, numPerFrame)
   data = readData(f, shape, numPerFrame)
   f.close()

