#ifndef COLUMNENERGYPROBETRIGGERCOMPONENT_HPP_
#define COLUMNENERGYPROBETRIGGERCOMPONENT_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/ProbeTriggerComponent.hpp"

namespace PV {

class ColumnEnergyProbeTriggerComponent : public ProbeTriggerComponent {
  protected:
   virtual void ioParam_reductionInterval(enum ParamsIOFlag);

  public:
   ColumnEnergyProbeTriggerComponent(char const *objName, PVParams *params);
   virtual ~ColumnEnergyProbeTriggerComponent() {}

   void initialize(char const *objName, PVParams *params);
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual bool needUpdate(double simTime, double deltaTime) override;

  private:
   int mReductionInterval = 0;
   int mReductionCounter  = 0;
    
};

// namespace PV

#endif // COLUMNENERGYPROBETRIGGERCOMPONENT_HPP_
