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
   NormalizeL3(const char * probeName, HyPerCol * hc);
   ~NormalizeL3();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int normalizeWeights();

protected:
   NormalizeL3();
   int initialize(const char * name, HyPerCol * hc);
   virtual void ioParam_minL3NormTolerated(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

   // Member variables
protected:
   float minL3NormTolerated; // Error if sqrt(sum(|weights|^3)) in any patch is less than this amount.
}; // end class NormalizeL3

BaseObject * createNormalizeL3(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif // NORMALIZEL3_HPP_
