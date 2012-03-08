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

   virtual int updateState(float timef, float dt);
   // virtual int updateV();
protected:
   int initialize();
   int updateState(float timef, float dt, int numNeurons, pvdata_t * V, pvdata_t * A, int nx, int ny, int nf, int nb);

   static int updateV_DatastoreDelayTestLayer(const PVLayerLoc * loc, bool * inited, pvdata_t * V, int period);

protected:
   bool inited;
   int period;  // The periodicity of the V buffer, in pixels.

}; // end of class DatastoreDelayTestLayer block

}  // end of namespace PV block

#endif /* DATASTOREDELAYTEST_HPP_ */
