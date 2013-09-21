/*
 * DatastoreDelayTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef DATASTOREDELAYTESTPROBE_HPP_
#define DATASTOREDELAYTESTPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <columns/HyPerCol.hpp>
#include <include/pv_common.h>

namespace PV {

class DatastoreDelayTestProbe: public StatsProbe {
public:
   DatastoreDelayTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);

   virtual ~DatastoreDelayTestProbe();

protected:
   int initDatastoreDelayTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg);

protected:
   char * name;
};

}

#endif /* DATASTOREDELAYTESTPROBE_HPP_ */
