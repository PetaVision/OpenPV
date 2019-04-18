/*
 * ImagePvpTestLayer.hpp
 * Author: slundquist
 */

#ifndef IMAGEPVPTESTLAYER_HPP_
#define IMAGEPVPTESTLAYER_HPP_
#include <layers/PvpLayer.hpp>

namespace PV {

class ImagePvpTestLayer : public PV::PvpLayer {
  public:
   ImagePvpTestLayer(const char *name, HyPerCol *hc);
   virtual Response::Status updateState(double time, double dt) override;

  protected:
   /**
    * In addition to base-class registerData, broadcast the fileCount to all processes.
    * All processes need to know the number of input images in order to verify that
    * they received the correct data, but ordinarily, only the parent class knows.
    */
   virtual Response::Status registerData(Checkpointer *checkpointer) override;

  private:
   int mNumFrames = 0; // The number of frames in the pvp file at the input path.
};
}

#endif // IMAGEPVPTESTLAYER_HPP_
