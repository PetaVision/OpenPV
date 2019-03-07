/*
 * NormalizeL3.hpp
 *
 * A probe to check that a layer is constant, with a value given by the parameter "correctValue"
 */

#ifndef NORMALIZEL3_HPP_
#define NORMALIZEL3_HPP_

#include <normalizers/NormalizeMultiply.hpp>

namespace PV {

class NormalizeL3 : public NormalizeMultiply {
  public:
   NormalizeL3(const char *name, PVParams *params, Communicator const *comm);
   ~NormalizeL3();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int normalizeWeights() override;

  protected:
   NormalizeL3();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void ioParam_minL3NormTolerated(enum ParamsIOFlag ioFlag);

   // Member variables
  protected:
   float minL3NormTolerated = 0.0f;
   // Error if sqrt(sum(|weights|^3)) in any patch is less than this amount.
}; // end class NormalizeL3

} // end namespace PV

#endif // NORMALIZEL3_HPP_
