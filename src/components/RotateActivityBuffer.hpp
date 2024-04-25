#ifndef ROTATEACTIVITYBUFFER_HPP_
#define ROTATEACTIVITYBUFFER_HPP_

#include "columns/Random.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "io/PVParams.hpp"
#include "structures/Buffer.hpp"
#include <memory>

namespace PV {

/**
 * A layer activity component to rotate around the center point.
 * Each time a layer update is triggered, it takes the data on the V buffer and rotates it by a
 * random angle. The random factor is taken from a uniform random distribution with min and
 * max specified by parameters angleMin and angleMax. The string parameter angleUnits is required
 * and takes the value "degrees" or "radians". 
 * Each batch element is rotated by a different angle, and the angles for different batch elements
 * are independent.
 */
class RotateActivityBuffer : public HyPerActivityBuffer {
  protected:
   /**
    * List of parameters used by the RotateActivityBuffer class
    * @name RotateLayer Parameters
    * @{
    */

   /**
    * @brief angleMin: The minimum possible value for the angle
    * Each display period, the angle is chosen randomly from
    * the interval [angleMin, angleMax].
    */
   virtual void ioParam_angleMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief angleMax: The maximum possible value for the angle
    * Each display period, the angle is chosen randomly from
    * the interval [angleMin, angleMax].
    * If the max and min are equal, the only possible choice is the common value.
    * If the max is less then the min, a warning is issued and the values are flipped.
    */
   virtual void ioParam_angleMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief angleUnits: The units for the angles in the angle file path.
    * Must be "degree", "degrees", "radian", or "radians" (case insensitive).
    */
   virtual void ioParam_angleUnits(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeAnglesFile: The path to the file where angles are recorded.
    * @details Set to NULL or the empty string if no such file is wanted.
    * The file is a text file, with the format
    *
    * t=[simulation time], b=[batch element], [angle in radians][linefeed]
    *
    * Note that the file gets written only by those processes that are both the root processes of
    * their I/O MPI block, and have row and column index equal to zero.
    *
    * The default is NULL (do not write to file).
    */
   virtual void ioParam_writeAnglesFile(enum ParamsIOFlag ioFlag);

   /** @} */

  public:
   RotateActivityBuffer(char const *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   enum class AngleUnitType { UNSET, DEGREES, RADIANS };

   RotateActivityBuffer();

   virtual Response::Status allocateDataStructures() override;
   void applyTransformCPU(
         Buffer<float> const &inputBuffer, Buffer<float> &outputBuffer, float angle);
   void initialize(char const *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   void copyRandStateToCheckpointData();
   void copyCheckpointDataToRandState();

   virtual Response::Status prepareCheckpointWrite(double simTime) override;
   virtual Response::Status processCheckpointRead(double simTime) override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual void setObjectType() override;
   void transform(Buffer<float> &localVBuffer, int bLocal, float angleRadians);

  protected:
   float mAngleConversionFactor = 1.0f;
   float mAngleMin;
   float mAngleMax;
   AngleUnitType mAngleUnitType = AngleUnitType::UNSET;
   std::vector<unsigned int> mRandStateCheckpointData;
   std::shared_ptr<Random> mRandState = nullptr;
   char *mWriteAnglesFile             = nullptr;

   // FileStream to output file used when mWriteAnglesFile is set
   std::shared_ptr<FileStream> mWriteAnglesStream;
};

} // namespace PV

#endif // ROTATEACTIVITYBUFFER_HPP_
