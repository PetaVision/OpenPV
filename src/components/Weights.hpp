/*
 * Weights.hpp
 *
 *  Created on: Jul 21, 2017
 *      Author: Pete Schultz
 */

#ifndef WEIGHTS_HPP_
#define WEIGHTS_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "components/PatchGeometry.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_types.h"
#include <memory>
#include <string>
#include <vector>

#ifdef PV_USE_CUDA
#include "arch/cuda/CudaDevice.hpp"
#endif // PV_USE_CUDA

namespace PV {

/**
 * Weights enpapsulates the patch geometry and patch data for a HyPerConn-type connection
 * It makes use of a PatchGeometry object (either creating it itself or using an already existing
 * object). It handles both shared and nonshared versions of the connection data used by HyPerConn,
 * and handles multiple arbors.
 *
 * If instantiating with the constructor that only takes the name argument, one of the initialize()
 * methods must be called. The initialize() methods differ in how the PatchGeometry object is
 * specified. Afterward, the allocateDataStructures() method must be called before the weights
 * object is ready. If the Weights is constructed with sharedWeights off, NumDataPatchesX and
 * NumDataPatchesY are the same as the PatchGeometry object's numPatchesX and numPatchesY.
 *
 * If sharedWeights is on, NumDataPatchesX is PreLoc.nx / PostLoc.nx if this quantity is
 * greater than one, and 1 otherwise. Note that the PatchGeometry object requires that
 * the quotient of PreLoc.nx and PostLoc.nx be an integral power of two (1, 2, 4, 8, ...; or
 * 1/2, 1/4, 1/8, ...).
 * NumDataPatchesY is defined similarly in terms of PreLoc.ny / PostLoc.ny.
 *
 * In both cases, numDataPatchesF is the same as the PatchGeometry object's NumPatchesF.
 */
class Weights {

  public:
   /**
    * Instantiates the Weights object and sets the name, but does not set any of the other
    * data members. One of the initialize() methods and then the allocateDataStructures()
    * method has to be called before the Weights object is ready to use.
    */
   Weights(std::string const &name);

   /**
    * Instantiates the Weights object and then calls the initialize() method with the arguments
    * after the name argument.
    */
   Weights(
         std::string const &name,
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         int numArbors,
         bool sharedWeights,
         double timestamp);

   /** The destructor for Weights. */
   virtual ~Weights() {}

   /**
    * An initializer that uses the specified PatchGeometry object as its patch geometry.
    * This method allows two Weights objects to share the same PatchGeometry object.
    */
   void initialize(
         std::shared_ptr<PatchGeometry> geometry,
         int numArbors,
         bool sharedWeights,
         double timestamp);

   /** A constructor that uses the geometry of an existing Weights object.
    * The PatchGeometry object is shared, and the other data members are copied.
    */
   void initialize(Weights const *baseWeights);

   /**
    * An initializer that takes the patch size and pre- and post-synaptic PVLayerLoc arguments
    * to define the PatchGeometry object.
    */
   void initialize(
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         int numArbors,
         bool sharedWeights,
         double timestamp);

   /**
    * Allocates the patch geometry and the the patch data vector.
    * The Weights object is not completely initialized until this method is called.
    * Once allocateDataStructures() is called once, subsequent calls return immediately
    * and have no effect.
    */
   void allocateDataStructures();

   void checkpointWeightPvp(Checkpointer *checkpointer, char const *bufferName, bool compressFlag);

   /** Calculates the minimum value of the patch data over all arbors. For nonshared weights, only
    * the active regions of the patches are considered when taking the minimum. */
   float calcMinWeight();

   /** Calculates the minimum value of the patch data over the given arbor. For nonshared weights,
    * only the active regions of the patches are considered when taking the minimum. */
   float calcMinWeight(int arbor);

   /** Calculates the maximum value of the patch data over all arbors. For nonshared weights, only
    * the active regions of the patches are considered when taking the minimum. */
   float calcMaxWeight();

   /** Calculates the maximum value of the patch data over the given arbor. For nonshared weights,
    * only the active regions of the patches are considered when taking the minimum. */
   float calcMaxWeight(int arbor);

   int calcDataIndexFromPatchIndex(int patchIndex) const;

#ifdef PV_USE_CUDA
   /**
    * If CUDA is being used, copy the weights onto the GPU.
    */
   void copyToGPU();
#endif // PV_USE_CUDA

   /** The get-method for the sharedWeights flag */
   bool getSharedFlag() const { return mSharedFlag; }

   /** The get-method for the name of the object */
   std::string const &getName() const { return mName; }

   /** The get-method for the PatchGeometry object */
   std::shared_ptr<PatchGeometry> getGeometry() const { return mGeometry; }

   /** The get-method for the number of arbors */
   int getNumArbors() const { return mNumArbors; }

   /**
    * The get-method for the number of data patches in the x-direction.
    * For shared weights, this is the number of kernels in the x-direction.
    * For nonshared weights, it is the number of neurons in the extended region in the x-direction.
    */
   int getNumDataPatchesX() const { return mNumDataPatchesX; }

   /**
    * The get-method for the number of data patches in the y-direction.
    * For shared weights, this is the number of kernels in the y-direction.
    * For nonshared weights, it is the number of neurons in the extended region in the y-direction.
    */
   int getNumDataPatchesY() const { return mNumDataPatchesY; }

   /**
    * The get-method for the number of data patches in the feature direction. For both shared and
    * nonshared weights, this is the number of features in the presynaptic layer.
    */
   int getNumDataPatchesF() const { return mNumDataPatchesF; }

   /** Returns the overall number of data patches */
   int getNumDataPatches() const {
      return getNumDataPatchesX() * getNumDataPatchesY() * getNumDataPatchesF();
   }

   /** Returns a nonmutable reference to the patch info for the given patch index. */
   Patch const &getPatch(int patchIndex) const;

   /** Returns a pointer to the patch data for the given arbor */
   float *getData(int arbor);

   /** Returns a read-only pointer to the patch data for the given arbor */
   float const *getDataReadOnly(int arbor) const;

   /** Returns a pointer to the patch data for the given data index */
   float *getDataFromDataIndex(int arbor, int dataIndex);

   /**
    * Returns a pointer to the patch data for data index corresponding to the
    * given patch index
    */
   float *getDataFromPatchIndex(int arbor, int patchIndex);

   /**
    * Returns a modifiable pointer to the patch data for the given arbor, and sets the timestamp to
    * the given value
    */
   float *getData(int arbor, double timestamp);

   /** Returns a modifiable pointer to the patch data for the given data index, and sets the
    * timestamp to the given value
    */
   float *getDataFromDataIndex(int arbor, int dataIndex, double timestamp);

   /** Returns a modifiable pointer to the patch data for data index corresponding to the given
    * patch index, and sets the timestamp to the give value
    */
   float *getDataFromPatchIndex(int arbor, int patchIndex, double timestamp);

   /** Sets the timestamp */
   void setTimestamp(double timestamp) { mTimestamp = timestamp; }

   /** Retrieves a previously set timestamp */
   double getTimestamp() const { return mTimestamp; }

   /** The get-method for the patch size in the x-dimension */
   int getPatchSizeX() const { return mGeometry->getPatchSizeX(); }

   /** The get-method for the patch size in the y-dimension */
   int getPatchSizeY() const { return mGeometry->getPatchSizeY(); }

   /** The get-method for the patch size in the feature dimension */
   int getPatchSizeF() const { return mGeometry->getPatchSizeF(); }

   /**
    * Returns getPatchSizeX() * getPatchSizeY() * getPatchSizeF(),
    * the overall number of items in a patch.
    */
   int getPatchSizeOverall() const { return mGeometry->getPatchSizeOverall(); }

   /**
    * Returns the memory stride between adjacent feature indices with the same x- and y- coordinates
    */
   int getPatchStrideF() const { return 1; }

   /**
    * Returns the memory stride between adjacent x-coordinates with the same y-coordinate and
    * feature index
    */
   int getPatchStrideX() const { return mGeometry->getPatchStrideX(); }

   /**
    * Returns the memory stride between adjacent y-coordinates with the same x-coordinate and
    * feature index
    */
   int getPatchStrideY() const { return mGeometry->getPatchStrideY(); }

   bool getWeightsArePlastic() const { return mWeightsArePlastic; }

   void setWeightsArePlastic() { mWeightsArePlastic = true; }

   /**
    * Copies the given halos into the halos of the PatchGeometry object.
    * in the PatchIt is an error to call this method after the PatchGeometry object's
    * allocateDataStructures method has been called.
    * (Recall that multiple Weights objects can share a PatchGeometry object).
    */
   void setMargins(PVHalo const &preHalo, PVHalo const &postHalo);

#ifdef PV_USE_CUDA
   bool isUsingGPU() { return mUsingGPUFlag; }
   void useGPU() { mUsingGPUFlag = true; }

   void setCudaDevice(PVCuda::CudaDevice *device) { mCudaDevice = device; }

   PVCuda::CudaBuffer *getDevicePatchToDataLookup() const { return mDevicePatchToDataLookup; }
   PVCuda::CudaBuffer *getDeviceData() const { return mDeviceData; }
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCUDNNData() const { return mCUDNNData; }
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

  protected:
   /**
    * The default constructor, called by derived classes (e.g. PoolingWeights).
    * Derived classes need to call Weights::initialize themselves
    */
   Weights() {}

   void setName(std::string const &name) { mName = name; }

   void setNumDataPatches(int numDataPatchesX, int numDataPatchesY, int numDataPatchesF);

#ifdef PV_USE_CUDA
   void allocateCudaBuffers();
#endif // PV_USE_CUDA

  private:
   virtual void initNumDataPatches();

  private:
   std::string mName;
   std::shared_ptr<PatchGeometry> mGeometry = nullptr;
   int mNumArbors;
   bool mSharedFlag;
   double mTimestamp;
   int mNumDataPatchesX;
   int mNumDataPatchesY;
   int mNumDataPatchesF;

   std::vector<std::vector<float>> mData;
   std::vector<int> dataIndexLookupTable;

   bool mWeightsArePlastic = false;

#ifdef PV_USE_CUDA
   bool mUsingGPUFlag                           = false;
   PVCuda::CudaDevice *mCudaDevice              = nullptr;
   PVCuda::CudaBuffer *mDevicePatchToDataLookup = nullptr;
   PVCuda::CudaBuffer *mDeviceData              = nullptr;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *mCUDNNData = nullptr;
#endif // PV_USE_CUDNN
   double mTimestampGPU;
#endif // PV_USE_CUDA
}; // end class Weights

} // end namespace PV

#endif // WEIGHTS_HPP_
