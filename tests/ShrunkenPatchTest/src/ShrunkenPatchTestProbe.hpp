/*
 * ShrunkenPatchTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef SHRUNKENPATCHTESTPROBE_HPP_
#define SHRUNKENPATCHTESTPROBE_HPP_

#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class ShrunkenPatchTestProbe : public PV::StatsProbeImmediate {
  public:
   ShrunkenPatchTestProbe(const char *probename, PVParams *params, Communicator const *comm);

   virtual ~ShrunkenPatchTestProbe();

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *probename, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nxpShrunken(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nypShrunken(enum ParamsIOFlag ioFlag);

  protected:
   int mNxpShrunken;
   int mNypShrunken;
   float *mCorrectValues;
}; // end class ShrunkenPatchTestProbe

} // end namespace PV

#endif /* SHRUNKENPATCHTESTPROBE_HPP_ */
