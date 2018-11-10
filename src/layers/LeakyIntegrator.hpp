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

class LeakyIntegrator : public ANNLayer {
   // Member functions
  public:
   LeakyIntegrator(const char *name, HyPerCol *hc);
   virtual ~LeakyIntegrator();

  protected:
   LeakyIntegrator();
   int initialize(const char *name, HyPerCol *hc);
   virtual ActivityComponent *createActivityComponent() override;

  private:
   int initialize_base();

   // Member Variables
  protected:
   float integrationTime;
}; // class LeakyIntegrator

} /* namespace PV */
#endif /* LEAKYINTEGRATOR_HPP_ */
