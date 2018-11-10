/*
 * FilenameParsingGroundTruthLayer.hpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */
#ifndef FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
#define FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_

#include "layers/HyPerLayer.hpp"

#include "components/InputLayerNameParam.hpp"
#include "layers/InputLayer.hpp"
#include <string>

namespace PV {

class FilenameParsingGroundTruthLayer : public HyPerLayer {
  public:
   FilenameParsingGroundTruthLayer(const char *name, HyPerCol *hc);
   virtual ~FilenameParsingGroundTruthLayer();
   virtual bool needUpdate(double simTime, double dt) const override;

  private:
   InputLayerNameParam *mInputLayerNameParam = nullptr;
   InputLayer *mInputLayer                   = nullptr;

  protected:
   virtual void createComponentTable(char const *description) override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual InputLayerNameParam *createInputLayerNameParam();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
}; // end class FilenameParsingGroundTruthLayer

} // end namespace PV

#endif // FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
