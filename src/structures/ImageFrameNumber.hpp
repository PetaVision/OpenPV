/*
 * ImageFrameNumber.hpp
 */

#ifndef IMAGEFRAMENUMBER_HPP_
#define IMAGEFRAMENUMBER_HPP_

#include <string>

namespace PV {

/**
 * A class to bundle a filename (for a file that contains one or more image frames),
 * with an integer-valued frame number. The frame number might be used to point to a
 * particular frame within the file, or the total number of frames in the file.
 */

class ImageFrameNumber {
  public:
   ImageFrameNumber() : mPath(""), mFrameNumber(0) {}
   ImageFrameNumber(std::string const &path, int frameNumber) :
         mPath(path), mFrameNumber(frameNumber) {}
   std::string const &getPath() const { return mPath; }
   int getFrameNumber() const { return mFrameNumber; }
  private:
   std::string mPath;
   int mFrameNumber;
};

} // namespace PV

#endif // IMAGEFRAMENUMBER_HPP_
