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
   FilenameParsingGroundTruthLayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~FilenameParsingGroundTruthLayer();

  private:
   InputLayerNameParam *mInputLayerNameParam = nullptr;

  protected:
   virtual void createComponentTable(char const *description) override;
   virtual LayerUpdateController *createLayerUpdateController();
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual InputLayerNameParam *createInputLayerNameParam();
}; // end class FilenameParsingGroundTruthLayer

} // end namespace PV

#endif // FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
