/*
 * CliqueLayer.hpp
 *
 *  Created on: Sep 3, 2011
 *      Author: gkenyon
 */

#ifndef CLIQUELAYER_HPP_
#define CLIQUELAYER_HPP_

#include "../columns/HyPerCol.hpp"
#include "../connections/HyPerConn.hpp"
#include "ANNLayer.hpp"


namespace PV {

class CliqueLayer: public PV::ANNLayer {
public:
   CliqueLayer(const char * name, HyPerCol * hc);
   virtual ~CliqueLayer();
   virtual int recvAllSynapticInput(); // Overrides since HyPerLayer::recvAllSynapticINput calls connections' deliver method, but when CliqueLayer was built it called layer recv methods
   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int axonId);
protected:
   CliqueLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Voffset(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vgain(enum ParamsIOFlag ioFlag);
   /* static */ int updateStateClique(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t Voffset, pvdata_t Vgain, pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, int columnID);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);
   virtual int recvSynapticInputBase(HyPerConn * conn, const PVLayerCube * activity, int arborID);
   pvdata_t Vgain;
   pvdata_t Voffset;
private:
   int initialize_base();
};

} /* namespace PV */

#endif /* CLIQUELAYER_HPP_ */
