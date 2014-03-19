/*
 * WindowTestLayer.hpp
 *
 *  Created on: Sep 27, 2011
 *      Author: gkenyon
 */

#ifndef WINDOWTESTLAYER_HPP_
#define WINDOWTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>
#include <layers/HyPerLCALayer.hpp>

namespace PV {

class WindowTestLayer: public PV::ANNLayer {
public:
   WindowTestLayer(const char* name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateState(double time, double dt);
   virtual int publish(InterColComm * comm, double timed);
   //int setVtoGlobalPos();
   int setActivitytoOne();

private:
    int initialize(const char * name, HyPerCol * hc);
    int initialize_base();
    //const char * windowLayerName;
    //HyPerLCALayer * windowLayer;
};

} /* namespace PV */
#endif /* WINDOWTESTLAYER_HPP_ */
