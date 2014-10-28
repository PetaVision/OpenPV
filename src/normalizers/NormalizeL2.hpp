/*
 * NormalizeL2.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEL2_HPP_
#define NORMALIZEL2_HPP_

#include "NormalizeMultiply.hpp"

namespace PV {

class NormalizeL2: public PV::NormalizeMultiply {
   // Member functions
   public:
      NormalizeL2(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConns);
      virtual ~NormalizeL2();

      virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
      virtual int normalizeWeights();

   protected:
      NormalizeL2();
      int initialize(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConns);

      virtual void ioParam_minL2NormTolerated(enum ParamsIOFlag ioFlag);

   private:
      int initialize_base();

   // Member variables
   protected:
      float minL2NormTolerated; // Error if sqrt(sum(weights^2)) in any patch is less than this amount.
   };

   } /* namespace PV */
#endif /* NORMALIZEL2_HPP_ */
