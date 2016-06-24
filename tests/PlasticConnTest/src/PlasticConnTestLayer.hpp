/*
 * PlasticConnTestLayer.hpp
 *
 *  Created on: Oct 24, 2011
 *      Author: pschultz
 */

#ifndef PLASTICCONNTESTLAYER_HPP_
#define PLASTICCONNTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class PlasticConnTestLayer: public PV::ANNLayer {
public:
   PlasticConnTestLayer(const char* name, HyPerCol * hc);
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   virtual int publish(InterColComm * comm, double timef);
protected:
   int copyAtoV();
   int setActivitytoGlobalPos();
   int initialize(const char * name, HyPerCol * hc);
}; // end class PlasticConnTestLayer

BaseObject * createPlasticConnTestLayer(char const * name, HyPerCol * hc);

} // end namespace PV
#endif /* PLASTICCONNTESTLAYER_HPP_ */
