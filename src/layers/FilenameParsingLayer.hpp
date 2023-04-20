/*
 * FilenameParsingLayer.hpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */
#ifndef FILENAMEPARSINGLAYER_HPP_
#define FILENAMEPARSINGLAYER_HPP_

#include "layers/HyPerLayer.hpp"

#include "components/InputLayerNameParam.hpp"
#include "layers/InputLayer.hpp"
#include <string>

namespace PV {

class FilenameParsingLayer : public HyPerLayer {
  public:
   FilenameParsingLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~FilenameParsingLayer();

  private:
   InputLayerNameParam *mInputLayerNameParam = nullptr;

  protected:
   virtual void fillComponentTable() override;
   virtual LayerUpdateController *createLayerUpdateController() override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual ActivityComponent *createActivityComponent() override;
   virtual InputLayerNameParam *createInputLayerNameParam();
}; // end class FilenameParsingLayer

} // end namespace PV

#endif // FILENAMEPARSINGLAYER_HPP_
