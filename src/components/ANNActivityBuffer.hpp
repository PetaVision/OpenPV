/*
 * ANNActivityBuffer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef ANNACTIVITYBUFFER_HPP_
#define ANNACTIVITYBUFFER_HPP_

#include "components/HyPerActivityBuffer.hpp"

#ifdef PV_USE_CUDA
#include "cudakernels/CudaUpdateANNActivity.hpp"
#endif

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class ANNActivityBuffer : public HyPerActivityBuffer {
  protected:
   /**
    * List of parameters used by the ANNActivityBuffer class
    * @name ANNLayer Parameters
    * @{
    */

   /**
    * @brief verticesV: An array of membrane potentials at points where the transfer function jumps
    * or changes slope.
    * There must be the same number of elements in verticesV as verticesA, and the sequence of
    * values must
    * be nondecreasing.  If this parameter is absent, layerListsVerticesInParams() returns false
    * and the vertices are computed internally from VThresh, AMin, AMax, AShift, and VWidth.
    * If the parameter is present, layerListsVerticesInParams() returns true.
    */
   virtual void ioParam_verticesV(enum ParamsIOFlag ioFlag);

   /**
    * @brief verticesA: An array of activities of points where the transfer function jumps or
    * changes slope.
    * There must be the same number of elements in verticesA as verticesV.
    * Only read if verticesV is present; otherwise it is computed internally from VThresh, AMin,
    * AMax, AShift, and VWidth.
    */
   virtual void ioParam_verticesA(enum ParamsIOFlag ioFlag);

   /**
    * @brief slopeNegInf: The slope of the transfer function when x is less than the first element
    * of verticesV.
    * Thus, if V < Vfirst, the corresponding value of A is A = Afirst - slopeNegInf * (Vfirst - V)
    * Only read if verticesV is present; otherwise it is computed internally from VThresh, AMin,
    * AMax, AShift, and VWidth.
    */
   virtual void ioParam_slopeNegInf(enum ParamsIOFlag ioFlag);

   /**
    * @brief slopePosInf: The slope of the transfer function when x is greater than the last element
    * of verticesV.
    * Thus, if V > Vlast, the corresponding value of A is A = Alast + slopePosInf * (V - Vlast)
    * Only read if verticesV is present; otherwise it is computed internally from VThresh, AMin,
    * AMax, AShift, and VWidth.
    */
   virtual void ioParam_slopePosInf(enum ParamsIOFlag ioFlag);

   /**
    * @brief VThresh: Only read if verticesV is absent.
    * The threshold value for the membrane potential.  Below this value, the
    * output activity will be AMin.  Above, it will obey the transfer function
    * as specified by AMax, VWidth, and AShift.  Default is -infinity.
    */
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);

   /**
    * @brief AMin: Only read if verticesV is absent.
    * When membrane potential V is below the threshold VThresh, activity
    * takes the value AMin.  Default is the value of VThresh.
    */
   virtual void ioParam_AMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief AMax: Only read if verticesV is absent.
    * Activity that would otherwise be greater than AMax is truncated to AMax.
    * Default is +infinity.
    */
   virtual void ioParam_AMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief AShift: Only read if verticesV is absent.
    * When membrane potential V is above the threshold VThresh, activity is V-AShift
    * (but see VWidth for making a gradual transition at VThresh).  Default is zero.
    */
   virtual void ioParam_AShift(enum ParamsIOFlag ioFlag);

   /**
    * @brief VWidth: Only read if verticesV is absent.
    * When the membrane potential is between VThresh and VThresh+VWidth, the activity changes
    * linearly
    * between A=AMin when V=VThresh and A=VThresh+VWidth-AShift when V=VThresh+VWidth.
    * Default is zero.
    */
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   ANNActivityBuffer(char const *name, PVParams *params, Communicator *comm);

   virtual ~ANNActivityBuffer();

   bool usingVerticesListInParams() const { return mVerticesListInParams; }

   float getVThresh() const { return mVThresh; }
   float getAMax() const { return mAMax; }
   float getAMin() const { return mAMin; }
   float getAShift() const { return mAShift; }
   float getVWidth() const { return mVWidth; }

  protected:
   ANNActivityBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status allocateDataStructures() override;

   /**
    * Called by allocateDataStructures to perform sanity checks on the
    * (V,A) vertices, and to allocate and compute the slopes.
    * If the params specified AMax, AMin, etc instead of verticesV, verticesA,
    * it also allocates and computes the (V,A) pairs that correspond to the
    * given params.
    */
   void allocateVerticesAndSlopes();

   /**
    * If the params file does not specify verticesV and verticesA explicitly,
    * ANNActivityBuffer::allocateDataStructures() calls this function to compute the vertices.
    */
   virtual void setVertices();

   /**
    * ANNActivityBuffer::allocateDataStructures() calls this function to perform sanity checking
    * on the vertices.  Returns PV_SUCCESS or PV_FAILURE to indicate whether the vertices
    * are acceptable.
    * For ANNActivityBuffer::checkVertices(), fails if the sequence of V vertices ever
    * decreases.  If the sequence of A vertices ever decreases, outputs a warning
    * but does not fail.
    */
   virtual void checkVertices() const;

   /**
    * ANNActivityBuffer::allocateDataStructures() calls this function to compute the slopes between
    * vertices.
    */
   void setSlopes();

   /**
    * Copies V to A, and then applies the parameters (either verticesV, verticesA, slopes; or
    * VThresh, AMax, AMin, AShift, VWidth) to the activity buffer.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   static void applyVerticesList(
         int nbatch,
         int numNeurons,
         float *A,
         float const *V,
         int nx,
         int ny,
         int nf,
         int lt,
         int rt,
         int dn,
         int up,
         int numVertices,
         float *verticesV,
         float *verticesA,
         float *slopes);

   static void applyVThresh(
         int nbatch,
         int numNeurons,
         float const *V,
         float AMin,
         float VThresh,
         float AShift,
         float VWidth,
         float *activity,
         int nx,
         int ny,
         int nf,
         int lt,
         int rt,
         int dn,
         int up);

   static void applyAMax(
         int nbatch,
         int numNeurons,
         float AMax,
         float *activity,
         int nx,
         int ny,
         int nf,
         int lt,
         int rt,
         int dn,
         int up);

#ifdef PV_USE_CUDA
   virtual void allocateUpdateKernel() override;

   virtual Response::Status copyInitialStateToGPU() override;

   virtual void updateBufferGPU(double simTime, double deltaTime) override;
#endif // PV_USE_CUDA

  protected:
   bool mVerticesListInParams = false;
   int mNumVertices           = 0;
   float *mVerticesV          = nullptr;
   float *mVerticesA          = nullptr;
   float *mSlopes             = nullptr;
   // slopes[0]=slopeNegInf; slopes[numVertices]=slopePosInf;
   // For k=1,...,numVertices-1, slopes[k]=slope from vertex k-1 to vertex k
   float mSlopeNegInf = 1.0f;
   float mSlopePosInf = 1.0f;

   float mVThresh = -FLT_MAX; // threshold potential, values smaller than VThresh are set to AMin
   float mAMax    = FLT_MAX; // maximum membrane potential, larger values are set to AMax
   float mAMin    = -FLT_MAX; // minimum membrane potential, smaller values are set to AMin
   float mAShift  = 0.0f; // shift potential, values above VThresh are shifted downward by AShift
   // AShift == 0, hard threshold condition
   // AShift == VThresh, soft threshold condition
   float mVWidth = 0.0f; // The thresholding occurs linearly over the region
// [VThresh,VThresh+VWidth].  VWidth=0,AShift=0 is standard hard-thresholding

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaVerticesV = nullptr;
   PVCuda::CudaBuffer *mCudaVerticesA = nullptr;
   PVCuda::CudaBuffer *mCudaSlopes    = nullptr;

   PVCuda::CudaUpdateANNActivity *mUpdateStateCudaKernel = nullptr;
#endif // PV_USE_CUDA
};

} // namespace PV

#endif // ANNACTIVITYBUFFER_HPP_
