/*
 * AssertZerosProbe.hpp
 * Author: slundquist
 */

#ifndef ASSERTZEROSPROBE_HPP_
#define ASSERTZEROSPROBE_HPP_
#include "probes/StatsProbe.hpp"

namespace PV {

class AssertZerosProbe : public PV::StatsProbe {
  public:
   AssertZerosProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();

}; // end class AssertZerosProbe

} // end namespace PV
#endif
