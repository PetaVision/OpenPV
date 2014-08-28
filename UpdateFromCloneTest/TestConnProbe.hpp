/*
 * TestConnProbe.hpp
 *
 */

#ifndef TESTCONNPROBE_HPP_
#define TESTCONNPROBE_HPP_

#include <io/BaseConnectionProbe.hpp>

namespace PV {

class TestConnProbe:BaseConnectionProbe {

// Methods
public:
   TestConnProbe(const char * probename, HyPerCol * hc);
   virtual ~TestConnProbe();
   virtual int outputState(double timed);

protected:
   TestConnProbe(); // Default constructor, can only be called by derived classes

private:
   int initialize_base();

};

}  // end of namespace PV block

#endif /* BASECONNECTIONPROBE_HPP_ */
