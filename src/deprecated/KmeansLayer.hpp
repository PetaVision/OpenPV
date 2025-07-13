/*
 * KmeansLayer.hpp
 *
 *  Created on: Dec. 1, 2014
 *      Author: Xinhua Zhang
 */

// KmeansLayer was deprecated on Aug 15, 2018.

#ifndef KMEANSLAYER_HPP_
#define KMEANSLAYER_HPP_

#include "layers/HyPerLayer.hpp"

namespace PV {
class KmeansLayer : public HyPerLayer {
  public:
   KmeansLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~KmeansLayer();

  protected:
   KmeansLayer();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual Response::Status updateState(double time, double dt) override;
   virtual int setActivity() override;
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void ioParam_TrainingFlag(ParamsIOSwitch ioSwitch);
   bool trainingFlag;

  private:
   int initialize_base();

}; // class KmeansLayer

} // namespace PV

#endif /* KMEANSLAYER_HPP_ */
