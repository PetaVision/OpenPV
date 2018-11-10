/*
 * ImagePvpTestBuffer.hpp
 * Author: slundquist
 */

#ifndef IMAGEPVPTESTBUFFER_HPP_
#define IMAGEPVPTESTBUFFER_HPP_
#include <components/PvpActivityBuffer.hpp>

namespace PV {

class ImagePvpTestBuffer : public PvpActivityBuffer {
  public:
   ImagePvpTestBuffer(const char *name, HyPerCol *hc);

  protected:
   /**
    * In addition to base-class registerData, broadcast the fileCount to all processes.
    * All processes need to know the number of input images in order to verify that
    * they received the correct data, but ordinarily, only the root process knows.
    */
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  private:
   int mNumFrames = 0; // The number of frames in the pvp file at the input path.
};
}

#endif // IMAGEPVPTESTBUFFER_HPP_
