/*
 * SpikingIntegrator.hpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#ifndef SPIKINGINTEGRATOR_HPP_
#define SPIKINGINTEGRATOR_HPP_

#include "ANNLayer.hpp"

namespace PV {

class SpikingIntegrator : public PV::ANNLayer {
   // Member functions
  public:
   SpikingIntegrator(const char *name, HyPerCol *hc);
   virtual ~SpikingIntegrator();

  protected:
   SpikingIntegrator();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_integrationTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vthresh(enum ParamsIOFlag ioFlag);
   virtual Response::Status updateState(double timed, double dt) override;

  private:
   int initialize_base();

   // Member Variables
  protected:
   float integrationTime;
   float Vthresh;
}; // class SpikingIntegrator

} /* namespace PV */
#endif /* SPIKINGINTEGRATOR_HPP_ */
