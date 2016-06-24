/*
 * ANNNormalizedErrorLayer.hpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#ifndef ANNNORMALIZEDERRORLAYER_HPP_
#define ANNNORMALIZEDERRORLAYER_HPP_

#include "ANNErrorLayer.hpp"
#include <fstream>
namespace PV {

class ANNNormalizedErrorLayer: public PV::ANNErrorLayer {
public:
   ANNNormalizedErrorLayer(const char * name, HyPerCol * hc);
   virtual ~ANNNormalizedErrorLayer();
   virtual double calcTimeScale(int batchIdx);
   virtual double getTimeScale(int batchIdx);
   virtual int updateState(double time, double dt);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   ANNNormalizedErrorLayer();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_useMask(enum ParamsIOFlag ioFlag);
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag);
   int initialize(const char * name, HyPerCol * hc);
private:
   int initialize_base();
   double* timeScale;
   std::ofstream timeScaleStream;

   bool useMask;
   char* maskLayerName;
   HyPerLayer* maskLayer;
}; // class ANNNormalizedErrorLayer

BaseObject * createANNNormalizedErrorLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* ANNNORMALIZEDERRORLAYER_HPP_ */
