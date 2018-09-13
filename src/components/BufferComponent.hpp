/*
 * BufferComponent.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef BUFFERCOMPONENT_HPP_
#define BUFFERCOMPONENT_HPP_

#include "columns/BaseObject.hpp"
#include "components/LayerGeometry.hpp"

namespace PV {

/**
 * A component to contain the phase parameter from the params file.
 */
class BufferComponent : public BaseObject {

  public:
   BufferComponent(char const *name, HyPerCol *hc);

   virtual ~BufferComponent();

   virtual void updateState(double simTime, double deltaTime) {}

   bool getExtendedFlag() const { return mExtendedFlag; }
   float const *getBufferData() const { return mBufferData.data(); }
   float const *getBufferData(int kBatch) const { return &mBufferData.at(kBatch * mBufferSize); }
   PVLayerLoc const *getLayerLoc() const { return mLayerGeometry->getLayerLoc(); }
   int getBufferSize() const { return mBufferSize; }
   int getBufferSizeAcrossBatch() const { return mBufferSize * getLayerLoc()->nbatch; }

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *getCudaBuffer() { return mCudaBuffer; }
   // TODO: eliminate need for nonconst public getCudaBuffer method

   void useCuda();
   void copyFromCuda();
   void copyToCuda();
#endif // PV_USE_CUDA

  protected:
   BufferComponent() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

  protected:
   bool mExtendedFlag = false;
   std::string mBufferLabel; // used in checkpointing to create the file name.
   LayerGeometry const *mLayerGeometry = nullptr;
   int mBufferSize                     = 0;
   std::vector<float> mBufferData;
   bool mInitializeFromCheckpointFlag;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaBuffer = nullptr;
#endif // PV_USE_CUDA
};

} // namespace PV

#endif // BUFFERCOMPONENT_HPP_
