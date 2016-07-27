/*
 * MomentumLCALayer.hpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#ifndef MOMENTUMLCALAYER_HPP_
#define MOMENTUMLCALAYER_HPP_

#include "HyPerLCALayer.hpp"

namespace PV {

class MomentumLCALayer: public PV::HyPerLCALayer{
public:
   MomentumLCALayer(const char * name, HyPerCol * hc);
   virtual ~MomentumLCALayer();
   virtual int checkpointWrite(const char * cpDir);
   virtual int checkpointRead(const char * cpDir, double * timeptr);

protected:
   MomentumLCALayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int allocateDataStructures();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * List of parameters needed from the MomentumLCALayer class
    * @name HyPerConn Parameters
    * @{
    */

   virtual void ioParam_LCAMomentumRate(enum ParamsIOFlag ioFlag);

   /** @} */

   virtual int updateState(double time, double dt);

#ifdef PV_USE_CUDA
   virtual int updateStateGpu(double time, double dt);
   virtual int allocateUpdateKernel();
#endif


   float LCAMomentumRate;
   pvdata_t * prevDrive;
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer* d_prevDrive;
#endif
   
private:
   int initialize_base();
}; // class MomentumLCALayer

} /* namespace PV */
#endif /* MOMENTUMLCALAYER_HPP_ */
