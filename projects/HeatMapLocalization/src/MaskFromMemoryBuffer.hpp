/*
 * MaskFromMemoryBuffer.hpp
 *
 *  Created on: Feb 18, 2016
 *      Author: peteschultz
 */

#ifndef MASKFROMMEMORYBUFFER_HPP_
#define MASKFROMMEMORYBUFFER_HPP_

#include <layers/ANNLayer.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>

class MaskFromMemoryBuffer: public PV::ANNLayer {
public:
   MaskFromMemoryBuffer(const char * name, PV::HyPerCol * hc);
   MaskFromMemoryBuffer();
   virtual ~MaskFromMemoryBuffer();
   virtual int communicateInitInfo();
protected:
   virtual int updateState(double time, double dt);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * @brief imageLayerName: the name of an ImageFromMemoryBuffer layer
    * @details The MaskFromMemoryBuffer layer will use imageLayer's
    * getImageLeft, getImageRight, getImageTop, getImageBottom member functions
    * to construct the mask.
    */
   virtual void ioParam_imageLayerName(enum ParamsIOFlag ioFlag);
private:
   int initialize_base();

// Member variables
   char* imageLayerName;
   PV::ImageFromMemoryBuffer * imageLayer;
   int imageLeft;
   int imageRight;
   int imageTop;
   int imageBottom;

};

#endif /* MASKFROMMEMORYBUFFER_HPP_ */
