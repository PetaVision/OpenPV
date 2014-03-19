/*
 * NormalizeScale.hpp (based on the code of NormalizeScale.cpp)
 *
 *  Created on: Mar 14, 2014
 *      Author: mpelko
 *
 * The name of normalize is misleading here. All this normalzition does is
 * multiply the weights by the strength parameter.
 *
 * Useful when doing Identity connection with non-one connection strength.
 * Usefull when you read the weights from a file and want to scale them
 * (without normalizations).
 */

#ifndef NORMALIZESCALE_HPP_
#define NORMALIZESCALE_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeScale: public PV::NormalizeBase {
   // Member functions
   public:
      NormalizeScale(HyPerConn * callingConn);
      virtual ~NormalizeScale();

      virtual int normalizeWeights(HyPerConn * conn);

   protected:
      NormalizeScale();
      int initialize(HyPerConn * callingConn);
      virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   private:
      int initialize_base();

   };

   } /* namespace PV */
#endif /* NORMALIZESCALE_HPP_ */
