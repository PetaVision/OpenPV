/*
 * ImageFromMemoryBuffer.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: Pete Schultz
 *  A subclass of BaseInput that processes an image based on an existing memory
 *  buffer instead of reading from a file.
 *
 *  Before using the image (typically after initializing the object but before
 *  calling the parent HyPerCol's run method), call the setMemoryBuffer() method.
 *  If using buildandrun, setMemoryBuffer() can be called using the custominit hook.
 */

// ImageFromMemoryBuffer broke under InputLayer refactoring and has not been maintained.
// It was deprecated on Aug 15, 2018 and marked obsolete on Nov 7, 2018.

#ifndef IMAGEFROMMEMORYBUFFER_HPP_
#define IMAGEFROMMEMORYBUFFER_HPP_

#include "layers/ImageLayer.hpp"

namespace PV {

class ImageFromMemoryBuffer : public ImageLayer {

  public:
   ImageFromMemoryBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~ImageFromMemoryBuffer();
}; // class ImageFromMemoryBuffer

} // namespace PV

#endif // IMAGEFROMMEMORYBUFFER_HPP_
