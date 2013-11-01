/*
 * BIDSCloneLayer.hpp
 * can be used to map BIDSLayers to larger dimensions
 *
 *  Created on: Jul 24, 2012
 *      Author: bnowers
 */

#ifndef BIDSCLONELAYER_HPP_
#define BIDSCLONELAYER_HPP_

#include "CloneVLayer.hpp"
#include "LIF.hpp"
#include "BIDSLayer.hpp"
#include "BIDSMovieCloneMap.hpp"

#include "../kernels/LIF_params.h"

namespace PV {

// TODO: Fix this comment for BIDS: CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class BIDSCloneLayer: public CloneVLayer {
public:
   BIDSCloneLayer(const char * name, HyPerCol * hc);
   virtual ~BIDSCloneLayer();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();
   int mapCoords();

protected:
   BIDSCloneLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int setParams(PVParams * params);
   virtual void readWriteSparseActivity(PVParams * params);
   virtual void readWriteSparseValues(PVParams * params);
   virtual void readJitterSource(PVParams * params);
   unsigned int * getSourceActiveIndices() {return originalLayer->getCLayer()->activeIndices;}
   unsigned int getSourceNumActive() {return originalLayer->getCLayer()->numActive;}
   int numNodes;
   BIDSCoords * coords;
   char * jitterSourceName;

private:
   int initialize_base();

};

}

#endif /* BIDSCLONELAYER_HPP_ */
