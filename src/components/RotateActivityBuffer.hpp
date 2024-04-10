#ifndef ROTATEACTIVITYBUFFER_HPP_
#define ROTATEACTIVITYBUFFER_HPP_

#include "columns/Random.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "io/PVParams.hpp"
#include "structures/Buffer.hpp"
#include <memory>

namespace PV {

class RotateActivityBuffer : public HyPerActivityBuffer {
  protected:
   /**
    * List of parameters used by the RotateActivityBuffer class
    * @name ANNLayer Parameters
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
   float interpolate(Buffer<float> const &inputBuffer, float xSrc, float ySrc, int feature);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void setObjectType() override;

  protected:
   float mAngleConversionFactor = 1.0f;
   float mAngleMin;
   float mAngleMax;
   AngleUnitType mAngleUnitType       = AngleUnitType::UNSET;
   std::shared_ptr<Random> mRandState = nullptr;
};

} // namespace PV

#endif // ROTATEACTIVITYBUFFER_HPP_
