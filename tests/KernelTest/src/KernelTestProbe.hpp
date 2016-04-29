/*
 * KernelTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef KERNELTESTPROBE_HPP_
#define KERNELTESTPROBE_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class KernelTestProbe: public PV::StatsProbe {
public:
   KernelTestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initKernelTestProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initKernelTestProbe_base();

};

BaseObject * createKernelTestProbe(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* KERNELTESTPROBE_HPP_ */
