/*
 * MPITestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef MPITESTPROBE_HPP_
#define MPITESTPROBE_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class MPITestProbe: public PV::StatsProbe {
public:
   MPITestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initMPITestProbe(const char * probeName, HyPerCol * hc);

private:
   int initMPITestProbe_base();
}; // end class MPITestProbe

BaseObject * createMPITestProbe(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* MPITESTPROBE_HPP_ */
