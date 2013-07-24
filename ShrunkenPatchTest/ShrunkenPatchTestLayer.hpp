/*
 * ShrunkenPatchTestLayer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef SHRUNKENPATCHTESTLAYER_HPP_
#define SHRUNKENPATCHTESTLAYER_HPP_

#include "../PetaVision/src/layers/ANNLayer.hpp"

namespace PV {

class ShrunkenPatchTestLayer: public PV::ANNLayer {
public:
   ShrunkenPatchTestLayer(const char* name, HyPerCol * hc, int numChannels);
   ShrunkenPatchTestLayer(const char* name, HyPerCol * hc);
   virtual int allocateDataStructures();
   virtual int updateState(double time, double dt);
   virtual int publish(InterColComm * comm, double timed);
   int setVtoGlobalPos();
   int setActivitytoGlobalPos();

private:
   int initialize(const char * name, HyPerCol * hc, int numChannels);

};

} /* namespace PV */
#endif /* SHRUNKENPATCHTESTLAYER_HPP_ */
