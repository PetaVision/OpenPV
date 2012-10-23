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
   virtual int updateState(double timef, double dt);
   virtual int updateActiveIndices();
protected:
   CliqueLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   /* static */ int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t Voffset, pvdata_t Vgain, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, int columnID);
   pvdata_t Vgain;
   pvdata_t Voffset;
   //int cliqueSize; // must get separately from each connection
private:
   int initialize_base();
};

} /* namespace PV */

#endif /* CLIQUELAYER_H_ */
