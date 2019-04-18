/*
 * LeakyIntegrator.hpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#ifndef LEAKYINTEGRATOR_HPP_
#define LEAKYINTEGRATOR_HPP_

#include "ANNLayer.hpp"

namespace PV {

class LeakyIntegrator : public PV::ANNLayer {
   // Member functions
  public:
   LeakyIntegrator(const char *name, HyPerCol *hc);
   virtual ~LeakyIntegrator();

  protected:
   LeakyIntegrator();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_integrationTime(enum ParamsIOFlag ioFlag);
   virtual Response::Status updateState(double timed, double dt) override;

  private:
   int initialize_base();

   // Member Variables
  protected:
   float integrationTime;
}; // class LeakyIntegrator

} /* namespace PV */
#endif /* LEAKYINTEGRATOR_HPP_ */
