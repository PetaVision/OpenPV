/*
 * BaseHyPerConnProbe.hpp
 *
 *  Created on: Oct 28, 2014
 *      Author: pschultz
 */

#ifndef BASEHYPERCONNPROBE_HPP_
#define BASEHYPERCONNPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/HyPerConn.hpp"

namespace PV {

class BaseHyPerConnProbe: public BaseConnectionProbe {
public:
   BaseHyPerConnProbe(const char * probeName, HyPerCol * hc);
   virtual ~BaseHyPerConnProbe();

   virtual int communicateInitInfo();

   HyPerConn * getTargetHyPerConn() { return targetHyPerConn; }

protected:
   BaseHyPerConnProbe();
   int initialize(const char * probeName, HyPerCol * hc);
   bool needRecalc(double timevalue);

private:
   int initialize_base();

protected:
   HyPerConn * targetHyPerConn;
};

} /* namespace PV */

#endif /* BASEHYPERCONNPROBE_HPP_ */
