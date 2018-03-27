/*
 * InitWeightTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef InitWeightTestProbe_HPP_
#define InitWeightTestProbe_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class InitWeightTestProbe : public PV::StatsProbe {
  public:
   InitWeightTestProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
