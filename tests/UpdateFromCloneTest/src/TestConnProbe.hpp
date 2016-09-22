/*
 * TestConnProbe.hpp
 *
 */

#ifndef TESTCONNPROBE_HPP_
#define TESTCONNPROBE_HPP_

#include "probes/BaseHyPerConnProbe.hpp"

namespace PV {

class TestConnProbe: public BaseHyPerConnProbe {

// Methods
public:
   TestConnProbe(const char * probename, HyPerCol * hc);
   virtual ~TestConnProbe();
   virtual int outputState(double timed);

protected:
   TestConnProbe(); // Default constructor, can only be called by derived classes
   
   /**
    * TestConnProbe::initNumValues() sets numValues to -1, indicating that getValues() and getValue() should not be used.
    */
   int initNumValues();
   
   /**
    * TestConnProbe::needRecalc() always returns false since calcValues should not be called.
    */
   bool needRecalc(double timevalue) { return false; }
   
   /**
    * TestConnProbe::calcValues() always fails since this probe does not use getValues() or getValue().
    */
   int calcValues(double timevalue) { return PV_FAILURE; }

private:
   int initialize_base();

}; // end of class TestConnProbe


}  // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
