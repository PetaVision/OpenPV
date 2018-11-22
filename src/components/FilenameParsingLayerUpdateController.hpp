/*
 * FilenameParsingLayerUpdateController.hpp
 *
 *  Created on: Nov 20, 2018
 *      Author: pschultz
 */

#ifndef FILENAMEPARSINGLAYERUPDATECONTROLLER_HPP_
#define FILENAMEPARSINGLAYERUPDATECONTROLLER_HPP_

#include "components/InputLayerUpdateController.hpp"

namespace PV {

/**
 * A component to determine if a layer should update on the current timestep, and to handle
 * triggering behavior.
 */
class FilenameParsingLayerUpdateController : public LayerUpdateController {
  public:
   FilenameParsingLayerUpdateController(char const *name, PVParams *params, Communicator *comm);
   virtual ~FilenameParsingLayerUpdateController();

   virtual bool needUpdate(double simTime, double deltaTime) const override;

  protected:
   FilenameParsingLayerUpdateController();

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   InputLayerUpdateController *mInputController = nullptr;

}; // class FilenameParsingLayerUpdateController

} // namespace PV

#endif // FILENAMEPARSINGLAYERUPDATECONTROLLER_HPP_
