#ifndef SCALEYACTIVITYBUFFER_HPP_
#define SCALEYACTIVITYBUFFER_HPP_

#include "columns/Random.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "io/PVParams.hpp"
#include "structures/Buffer.hpp"
#include <memory>

namespace PV {

/**
 * A layer activity component to stretch or contract in the y-direction.
 * Each time a layer update is triggered, it takes the data on the V buffer and stretches it by a
 * random factor. The random factor is taken from a uniform random distribution with min and max
 * specified by parameters scaleFactorMin and scaleFactorMax. Positive or negative values are
 * allowed, but the interval cannot intersect the range [-0.01, 0.01].
 * Each batch element is scaled by a different factor, and the factors for different batch elements
 * are independent.
 */
class ScaleYActivityBuffer : public HyPerActivityBuffer {
  protected:
   /**
    * List of parameters used by the ScaleYActivityBuffer class
    * @name ScaleYLayer Parameters
    * @{
    */

   /**
    * @brief scaleFactorMin: The minimum possible value for the scale factor
    * Each display period, the scale factor is chosen randomly from
    * the interval [scaleFactorMin, scaleFactorMax].
    */
   virtual void ioParam_scaleFactorMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief scaleFactorMax: The maximum possible value for the scale factor
    * Each display period, the scale factor is chosen randomly from
    * the interval [scaleFactorMin, scaleFactorMax].
    * If the max and min are equal, the only possible choice is the common value.
    * If the max is less then the min, a warning is issued and the values are flipped.
    */
   virtual void ioParam_scaleFactorMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeScaleFactorsFile: The path to the file where scale factors are recorded.
    * @details Set to NULL or the empty string if no such file is wanted.
    * The file is a text file, with the format
    *
    * t=[simulation time], b=[batch element], [scale factor][linefeed]
    *
    * Note that the file gets written only by those processes that are both the root processes of
    * their I/O MPI block, and have row and column index equal to zero.
    *
    * The default is NULL (do not write to file).
    */
   virtual void ioParam_writeScaleFactorsFile(enum ParamsIOFlag ioFlag);

   /** @} */

  public:
   ScaleYActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   ScaleYActivityBuffer();

   virtual Response::Status allocateDataStructures() override;
   void applyTransformCPU(
         Buffer<float> const &inputBuffer, Buffer<float> &outputBuffer, float scaleFactor);
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   void copyRandStateToCheckpointData();
   void copyCheckpointDataToRandState();

   virtual Response::Status prepareCheckpointWrite(double simTime) override;
   virtual Response::Status processCheckpointRead(double simTime) override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual void setObjectType() override;
   void transform(Buffer<float> &localVBuffer, int bLocal, float scaleFactor);

  protected:
   float mScaleFactorMin;
   float mScaleFactorMax;
   std::vector<unsigned int> mRandStateCheckpointData;
   std::shared_ptr<Random> mRandState = nullptr;
   char *mWriteScaleFactorsFile       = nullptr;

   // FileStream to output file used when mWriteAnglesFile is set
   std::shared_ptr<FileStream> mWriteScaleFactorsStream;
};

} // namespace PV

#endif // SCALEYACTIVITYBUFFER_HPP_
