/*
 * ComponentBuffer.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef COMPONENTBUFFER_HPP_
#define COMPONENTBUFFER_HPP_

#include "columns/BaseObject.hpp"

#include "components/LayerGeometry.hpp"
#ifdef PV_USE_CUDA
#include "arch/cuda/CudaKernel.hpp"
#endif // PV_USE_CUDA

namespace PV {

/**
 * The base class for layer buffers such as GSyn, membrane potential, activity, etc.
 */
class ComponentBuffer : public BaseObject {

  public:
   ComponentBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~ComponentBuffer();

   /**
    * Allows the buffer label to be set after instantiation.
    * Must be called before RegisterData for the label to have any effect.
    * If the buffer label has already been set to a nonempty string, this
    * method causes a fatal error.
    */
   void setBufferLabel(std::string const &label);

   void setBufferLabel(char const *label) { setBufferLabel(std::string(label)); }

   /**
    * The public method to update the buffer. Based on the value of UsingGPUFlag,
    * it calls either updateBufferCPU or updateBufferGPU. It also updates
    * the TimeLastUpdate value with the simTime argument.
    */
   void updateBuffer(double simTime, double deltaTime);

   bool getExtendedFlag() const { return mExtendedFlag; }
   int getNumChannels() const { return mNumChannels; }
   std::string const &getBufferLabel() const { return mBufferLabel; }
   bool getCheckpointFlag() const { return mCheckpointFlag; }

   /**
    * Returns a read-only pointer to the buffer's data.
    */
   float const *getBufferData() const { return dataPointer(0); }

   /**
    * Returns a read-only pointer to the the start of the specified batch (within channel zero).
    * Returns the null pointer if the needed offset is out of bounds for the buffer.
    */
   float const *getBufferData(int kBatch) const { return dataPointer(kBatch * mBufferSize); }

   /**
    * Returns a read-only pointer to the start of the specified batch within the specified channel
    * Returns the null pointer if the needed offset is out of bounds for the buffer.
    */
   float const *getBufferData(int kBatch, int channel) const {
      return dataPointer(channel * getBufferSizeAcrossBatch() + kBatch * mBufferSize);
   }

   /**
    * Returns a read-only pointer to the start of the specified channel.
    * Returns the null pointer if the channel is out of bounds.
    */
   float const *getChannelData(int channel) const {
      return dataPointer(channel * getBufferSizeAcrossBatch());
   }

   /**
    * Returns a pointer to the buffer's data that allows write access.
    * If this method returns the null pointer, the buffer should be treated as read-only.
    */
   float *getReadWritePointer() { return mReadWritePointer; }

   PVLayerLoc const *getLayerLoc() const { return mLayerGeometry->getLayerLoc(); }
   int getBufferSize() const { return mBufferSize; }
   int getBufferSizeAcrossBatch() const { return mBufferSizeAcrossBatch; }
   int getBufferSizeAcrossChannels() const { return mBufferSizeAcrossChannels; }

   double getTimeLastUpdate() const { return mTimeLastUpdate; }

   // A static method to check whether two buffers have compatible sizes
   static void checkDimensionsEqual(ComponentBuffer const *buffer1, ComponentBuffer const *buffer2);
   static void
   checkDimensionsXYEqual(ComponentBuffer const *buffer1, ComponentBuffer const *buffer2);
   static void checkBatchWidthEqual(ComponentBuffer const *buffer1, ComponentBuffer const *buffer2);

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *getCudaBuffer() { return mCudaBuffer; }
   // TODO: eliminate need for nonconst public getCudaBuffer method

   void useCuda();
   void copyFromCuda();
   void copyToCuda();
#endif // PV_USE_CUDA

  protected:
   ComponentBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   /**
    * Sets the values returned by the getBufferSize(), getBufferSizeAcrossBatch(),
    * and getBufferSizeAcrossChannels() methods.
    */
   void setBufferSize();

   /**
    * The method, called by allocateDataStructures, to set the pointer returned by
    * getReadOnlyPointer(). The default allocates the buffer data and sets the pointer to the
    * start of the data buffer. It can be overridden (e.g. by clones), however, it should always
    * set the pointer to a buffer whose size is at least BufferSizeAcrossChannels.
    */
   virtual void setReadOnlyPointer();

   /**
    * The method, called by allocateDataStructures, to set the pointer returned by
    * getBufferData(). The default returns the pointer to the start of the data buffer,
    * if the data buffer is non-empty, but the null pointer if it is empty (e.g. for
    * cloned membrane potentials, which use another layer's data buffer instead.)
    */
   virtual void setReadWritePointer();
#ifdef PV_USE_CUDA
   void setCudaBuffer();
#endif // PV_USE_CUDA

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   /**
    * The virtual method for updating the buffer data if not using the GPU for this buffer.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) {}

#ifdef PV_USE_CUDA
   virtual void allocateUpdateKernel() {}
   virtual void updateBufferGPU(double simTime, double deltaTime);
   virtual Response::Status copyInitialStateToGPU() override;
#endif // PV_USE_CUDA

  private:
   float const *dataPointer(int index) const {
      return (index >= 0 && index < mBufferSizeAcrossChannels) ? &mReadOnlyPointer[index] : nullptr;
   }

  protected:
   bool mExtendedFlag   = false;
   int mNumChannels     = 1;
   bool mCheckpointFlag = true; // Derived class can set this to false to suppress checkpointing.
   // See the comments on mBufferLabel for details.

   LayerGeometry const *mLayerGeometry = nullptr;
   int mBufferSize                     = 0;
   int mBufferSizeAcrossBatch          = 0;
   int mBufferSizeAcrossChannels       = 0;
   std::vector<float> mBufferData;
   float const *mReadOnlyPointer = nullptr;
   float *mReadWritePointer      = nullptr;

  private:
   // ComponentBuffer initializes mBuffferLabel to the empty string.
   // Some derived classes set mBufferlabel during instantiation. If mBufferLabel is not set
   // during instantiation, it can be set by calling setBufferLabel(). Note, however, that
   // setBufferLabel() cannot be called once the BufferLabel is set.
   // If mCheckpointFlag is true, the buffer is checkpointed with filename of the form
   // "[name]_[label].pvp". Note that if label is empty, the file will not be checkpointed
   // even if mCheckpointFlag is true.
   std::string mBufferLabel;
   double mTimeLastUpdate = 0.0;
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaBuffer = nullptr;
#endif // PV_USE_CUDA
};

} // namespace PV

#endif // COMPONENTBUFFER_HPP_
