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
   LeakyIntegrator(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~LeakyIntegrator();

  protected:
   LeakyIntegrator();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ActivityComponent *createActivityComponent() override;
}; // class LeakyIntegrator

} /* namespace PV */
#endif /* LEAKYINTEGRATOR_HPP_ */
