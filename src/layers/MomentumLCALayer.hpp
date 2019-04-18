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

class MomentumLCALayer : public PV::HyPerLCALayer {
  public:
   MomentumLCALayer(const char *name, HyPerCol *hc);
   virtual ~MomentumLCALayer();

  protected:
   MomentumLCALayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual Response::Status allocateDataStructures() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * List of parameters needed from the MomentumLCALayer class
    * @name HyPerConn Parameters
    * @{
    */

   virtual void ioParam_LCAMomentumRate(enum ParamsIOFlag ioFlag);

   /** @} */

   virtual Response::Status registerData(Checkpointer *checkpointer) override;
   virtual Response::Status processCheckpointRead() override;
   virtual Response::Status prepareCheckpointWrite() override;

   virtual Response::Status updateState(double time, double dt) override;

#ifdef PV_USE_CUDA
   virtual Response::Status updateStateGpu(double time, double dt) override;
   virtual int allocateUpdateKernel() override;
#endif

   float LCAMomentumRate;
   float *prevDrive;
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *d_prevDrive;
#endif

  private:
   int initialize_base();
}; // class MomentumLCALayer

} /* namespace PV */
#endif /* MOMENTUMLCALAYER_HPP_ */
