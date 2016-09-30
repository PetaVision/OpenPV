/*
 * MomentumTestConnProbe.hpp
 *
 */

#ifndef MOMENTUMTESTCONNPROBE_HPP_
#define MOMENTUMTESTCONNPROBE_HPP_

#include "probes/BaseHyPerConnProbe.hpp"

namespace PV {

class MomentumTestConnProbe : public BaseHyPerConnProbe {

   // Methods
  public:
   MomentumTestConnProbe(const char *probename, HyPerCol *hc);
   virtual ~MomentumTestConnProbe();
   virtual int outputState(double timed);

  protected:
   MomentumTestConnProbe(); // Default constructor, can only be called by derived classes

   /**
    * MomentumTestConnProbe::initNumValues() sets numValues to -1, indicating that getValues() and
    * getValue() should not be used.
    */
   int initNumValues();

   /**
    * MomentumTestConnProbe::needRecalc() always returns false since calcValues should not be
    * called.
    */
   bool needRecalc(double timevalue) { return false; }

   /**
    * MomentumTestConnProbe::calcValues() always fails since this probe does not use getValues() or
    * getValue().
    */
   int calcValues(double timevalue) { return PV_FAILURE; }

  private:
   int initialize_base();

}; // end of class MomentumTestConnProbe

} // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
