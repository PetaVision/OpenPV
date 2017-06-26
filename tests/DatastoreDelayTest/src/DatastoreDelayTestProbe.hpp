/*
 * DatastoreDelayTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef DATASTOREDELAYTESTPROBE_HPP_
#define DATASTOREDELAYTESTPROBE_HPP_

#include "columns/HyPerCol.hpp"
#include "include/pv_common.h"
#include "probes/StatsProbe.hpp"

namespace PV {

class DatastoreDelayTestProbe : public StatsProbe {
  public:
   DatastoreDelayTestProbe(const char *probename, HyPerCol *hc);

   virtual int outputState(double timed) override;

   virtual ~DatastoreDelayTestProbe();

  protected:
   int initDatastoreDelayTestProbe(const char *probename, HyPerCol *hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   virtual int communicateInitInfo(CommunicateInitInfoMessage const *message) override;

   // Data members
  private:
   int mNumDelayLevels = 0;
};
}

#endif /* DATASTOREDELAYTESTPROBE_HPP_ */
