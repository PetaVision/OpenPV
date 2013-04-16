/*
 * NormalizeL2.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEL2_HPP_
#define NORMALIZEL2_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeL2: public PV::NormalizeBase {
   // Member functions
   public:
      NormalizeL2(const char * name, PVParams * params);
      virtual ~NormalizeL2();

      virtual int normalizeWeights(HyPerConn * conn);

   protected:
      NormalizeL2();
      int initialize(const char * name, PVParams * params);
      virtual int setParams();

      virtual void readMinL2NormTolerated() {minL2NormTolerated = params->value(name, "minL2NormTolerated", 0.0f, true/*warnIfAbsent*/);}

   private:
      int initialize_base();

   // Member variables
   protected:
      float minL2NormTolerated; // Error if sqrt(sum(weights^2)) in any patch is less than this amount.
   };

   } /* namespace PV */
#endif /* NORMALIZEL2_HPP_ */
