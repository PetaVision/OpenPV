/*
 * DatastoreDelayTest.hpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#ifndef DATASTOREDELAYTEST_HPP_
#define DATASTOREDELAYTEST_HPP_

#include "../PetaVision/src/layers/ANNLayer.hpp"
#include "../PetaVision/src/utils/pv_random.h"

namespace PV {

class DatastoreDelayTestLayer : public ANNLayer {

public:
   DatastoreDelayTestLayer(const char* name, HyPerCol * hc);
   virtual ~DatastoreDelayTestLayer();

   virtual int updateV();
protected:
   int initialize();

protected:
   bool inited;
   int period;  // The periodicity of the V buffer, in pixels.

}; // end of class DatastoreDelayTestLayer block

}  // end of namespace PV block

#endif /* DATASTOREDELAYTEST_HPP_ */
