/*
 * HyPerLCALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef HYPERLCALAYER_HPP_
#define HYPERLCALAYER_HPP_

#include "ANNLayer.hpp"
#include "../io/SparsityLayerProbe.hpp"

namespace PV {

class HyPerLCALayer: public PV::ANNLayer {
public:
   HyPerLCALayer(const char * name, HyPerCol * hc);
   virtual ~HyPerLCALayer();

   //Overwriting HyPerLayer's window methods
   virtual int getNumWindows();
   virtual bool inWindowExt(int windowId, int neuronIdxExt);
   virtual bool inWindowRes(int windowId, int neuronIdxRes);
   //calcWindow used in WindowSystemTest, so needs to be public
   int calcWindow(int globalExtX, int globalExtY);

protected:
   HyPerLCALayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int allocateDataStructures();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numChannels(enum ParamsIOFlag ioFlag);
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numWindowX(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numWindowY(enum ParamsIOFlag ioFlag);
   virtual void ioParam_windowSymX(enum ParamsIOFlag ioFlag);
   virtual void ioParam_windowSymY(enum ParamsIOFlag ioFlag);
   virtual void ioParam_selfInteract(enum ParamsIOFlag ioFlag);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);

#ifdef PV_USE_CUDA
   virtual int doUpdateStateGpu(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead);
#endif

   virtual float getChannelTimeConst(enum ChannelType channel_type){return timeConstantTau;};

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int allocateUpdateKernel();
#endif
private:
   SparsityLayerProbe* sparseProbe;
   pvdata_t timeConstantTau;
   int initialize_base();
   int numWindowX;
   int numWindowY;
   bool windowSymX;
   bool windowSymY;
   bool selfInteract;
};

} /* namespace PV */
#endif /* HYPERLCALAYER_HPP_ */
