/*
 * CliqueLayer.h
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#ifndef CLIQUELAYER_H_
#define CLIQUELAYER_H_

#include "../columns/HyPerCol.hpp"
#include "../connections/HyPerConn.hpp"
#include "ANNLayer.hpp"


namespace PV {

class CliqueLayer: public PV::ANNLayer {
public:
   CliqueLayer(const char * name, HyPerCol * hc, int numChannels);
   CliqueLayer(const char * name, HyPerCol * hc);
   virtual ~CliqueLayer();
   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int axonId);
   virtual int updateState(float timef, float dt);
   virtual int updateActiveIndices();
protected:
   CliqueLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   /* static */ int updateState(float timef, float dt, int num_neurons, pvdata_t * V, pvdata_t * gSynExc, pvdata_t * gSynInh, pvdata_t * gSynInhB, pvdata_t Voffset, pvdata_t Vgain, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, int columnID);
   pvdata_t Vgain;
   pvdata_t Voffset;
   //int cliqueSize; // must get separately from each connection
private:
   int initialize_base();
};

} /* namespace PV */

#endif /* CLIQUELAYER_H_ */
