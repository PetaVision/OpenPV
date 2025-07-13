/*
 * ShrunkenPatchTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef SHRUNKENPATCHTESTPROBE_HPP_
#define SHRUNKENPATCHTESTPROBE_HPP_

#include <columns/Communicator.hpp>
#include <params/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class ShrunkenPatchTestProbe : public PV::StatsProbeImmediate {
  public:
   ShrunkenPatchTestProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~ShrunkenPatchTestProbe();

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void ioParam_nxpShrunken(ParamsIOSwitch ioSwitch);
   virtual void ioParam_nypShrunken(ParamsIOSwitch ioSwitch);

  protected:
   int mNxpShrunken;
   int mNypShrunken;
   float *mCorrectValues;
}; // class ShrunkenPatchTestProbe

} // namespace PV

#endif /* SHRUNKENPATCHTESTPROBE_HPP_ */
