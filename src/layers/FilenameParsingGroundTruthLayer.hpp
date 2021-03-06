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
   FilenameParsingGroundTruthLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~FilenameParsingGroundTruthLayer();

  private:
   InputLayerNameParam *mInputLayerNameParam = nullptr;

  protected:
   virtual void fillComponentTable() override;
   virtual LayerUpdateController *createLayerUpdateController() override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual InputLayerNameParam *createInputLayerNameParam();
}; // end class FilenameParsingGroundTruthLayer

} // end namespace PV

#endif // FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
