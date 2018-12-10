/*
 * LayerOutputComponent.hpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#ifndef LAYEROUTPUTCOMPONENT_HPP_
#define LAYEROUTPUTCOMPONENT_HPP_

#include "columns/BaseObject.hpp"

#include "checkpointing/CheckpointableFileStream.hpp"
#include "components/BasePublisherComponent.hpp"
#include "utils/Timer.hpp"

namespace PV {

/**
 * A component to output a layer's activity. It creates a .pvp file whose name is the
 * component's name followed by the suffix ".pvp" in the output directory.
 * It then responds to the LayerOutputStateMessage by appending a frame to the .pvp file
 * with the current activity state (retrieved from the publisher).
 * The boolean parameter sparseLayer controls whether the layer is sparse or dense.
 */
class LayerOutputComponent : public BaseObject {
  protected:
   /**
    * List of parameters needed from the LayerOutputComponent class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief writeStep: Specifies how often to output a pvp file for this layer
    * @details Defaults to every timestep. -1 specifies not to write at all.
    */
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

   /**
    * @brief initialWriteTime: Specifies the first timestep to start outputing pvp files
    */
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);

   /** @} */ // end of LayerOutputComponent parameters

  public:
   LayerOutputComponent(char const *name, PVParams *params, Communicator *comm);
   virtual ~LayerOutputComponent();

  protected:
   LayerOutputComponent();

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual void initMessageActionMap() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message);

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);

   int openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   virtual Response::Status outputState(double simTime, double deltaTime);

   /**
    * Appends the current activity to the OutputStateStream if SparseLayer is false.
    * Called by outputState.
    */
   virtual void writeActivity(double simTime, PVLayerCube &cube);

   /**
    * Appends the current activity to the OutputStateStream if SparseLayer is true.
    * Called by outputState.
    */
   virtual void writeActivitySparse(double simTime, PVLayerCube &cube);

   void updateNBands(int numCalls);

  protected:
   double mInitialWriteTime = 0.0; // time of first output
   double mWriteTime        = 0.0; // time of next output
   double mWriteStep        = 0.0; // output time interval

   BasePublisherComponent *mPublisher           = nullptr;
   CheckpointableFileStream *mOutputStateStream = nullptr; // file stream for the pvp file
   int mWriteActivityCalls; // Number of calls to writeActivity (written to nbands in the pvp file)
   int mWriteActivitySparseCalls; // Number of calls to writeActivitySparse (written to nbands in
   // the pvp file)

   Timer *mIOTimer = nullptr;
}; // class LayerOutputComponent

} // namespace PV

#endif // LAYEROUTPUTCOMPONENT_HPP_
