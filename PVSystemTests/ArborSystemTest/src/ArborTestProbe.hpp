/*
 * ArborTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ArborTestProbe_HPP_
#define ArborTestProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class ArborTestProbe: public PV::StatsProbe {
public:
   ArborTestProbe(const char * probeName, HyPerCol * hc);
   virtual ~ArborTestProbe();

   virtual int outputState(double timed);

protected:
   int initArborTestProbe(const char * probeName, HyPerCol * hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initArborTestProbe_base();

};

BaseObject * createArborTestProbe(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
