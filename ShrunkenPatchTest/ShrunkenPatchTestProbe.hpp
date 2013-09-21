/*
 * customStatsProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef SHRUNKENPATCHTESTPROBE_HPP_
#define SHRUNKENPATCHTESTPROBE_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class PVParams;

class ShrunkenPatchTestProbe: public PV::StatsProbe {
public:
   ShrunkenPatchTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg);
   ShrunkenPatchTestProbe(const char * probename, HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);

   virtual ~ShrunkenPatchTestProbe();

protected:
   int initShrunkenPatchTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg);
   virtual void readNxpShrunken(PVParams * params);
   virtual void readNypShrunken(PVParams * params);

protected:
   char * probeName;
   int nxpShrunken;
   int nypShrunken;
   pvdata_t * correctValues;
};

}

#endif /* SHRUNKENPATCHTESTPROBE_HPP_ */
