#ifndef IMAGEPVPOFFSETTESTLAYER_HPP_ 
#define IMAGEPVPOFFSETTESTLAYER_HPP_

#include <layers/ImagePvp.hpp>
#include <cMakeHeader.h>

namespace PV {

class ImagePvpOffsetTestLayer: public PV::ImagePvp{
public:
   ImagePvpOffsetTestLayer(const char* name, HyPerCol * hc);
   virtual double getDeltaUpdateTime();

protected:
   int updateState(double timef, double dt);

private:
};

BaseObject * createImagePvpOffsetTestLayer(const char * name, HyPerCol * hc);

} /* namespace PV */
#endif // IMAGEPVPOFFSETTESTLAYER_HPP_
