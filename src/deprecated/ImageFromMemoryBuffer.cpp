/*
 * ImageFromMemoryBuffer.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: Pete Schultz
 *  Description of the class is in ImageFromMemoryBuffer.hpp
 */

// ImageFromMemoryBuffer was deprecated on Aug 15, 2018.

#include "ImageFromMemoryBuffer.hpp"

#include <vector>

namespace PV {

// ImageFromMemoryBuffer was deprecated on Aug 15, 2018 and marked obsolete on Nov 7, 2018.
ImageFromMemoryBuffer::ImageFromMemoryBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   Fatal() << "Unable to create layer \"" << name << "\": ImageFromMemoryBuffer is obsolete.\n";
}

ImageFromMemoryBuffer::~ImageFromMemoryBuffer() {}

} // namespace PV
