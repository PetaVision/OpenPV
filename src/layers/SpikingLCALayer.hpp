/*
 * SpikingLCALayer.hpp
 *
 *  Created on: July 11, 2016
 *      Author: athresher
 */

#ifndef SPIKINGLCALAYER_HPP_
#define SPIKINGLCALAYER_HPP_

#include "ANNLayer.hpp"

namespace PV 
{
   class SpikingLCALayer: public PV::ANNLayer 
   {
      public:
         SpikingLCALayer(const char * name, HyPerCol * hc);
         virtual ~SpikingLCALayer();
         virtual double getDeltaUpdateTime();
         virtual int requireChannel(int channelNeeded, int * numChannelsResult) ;

      protected:
         SpikingLCALayer();
         int initialize(const char * name, HyPerCol * hc);
         virtual int allocateDataStructures();
         virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
         virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);
         virtual void ioParam_refactoryScale(enum ParamsIOFlag ioFlag);
         virtual int updateState(double time, double dt);
         virtual float getChannelTimeConst(enum ChannelType channel_type){return timeConstantTau;};

         pvdata_t timeConstantTau;
         float refactoryScale;

#ifdef PV_USE_CUDA
         virtual int updateStateGpu(double time, double dt);
         virtual int allocateUpdateKernel();
         PVCuda::CudaBuffer* d_dtAdapt;
         PVCuda::CudaBuffer* d_verticesV;
         PVCuda::CudaBuffer* d_verticesA;
         PVCuda::CudaBuffer* d_slopes;
#endif

      private:
         int initialize_base();
   };

   BaseObject * createSpikingLCALayer(char const * name, HyPerCol * hc);

}

#endif
