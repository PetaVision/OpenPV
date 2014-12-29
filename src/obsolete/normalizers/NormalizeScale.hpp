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

#include "NormalizeMultiply.hpp"

namespace PV {

class NormalizeScale: public PV::NormalizeMultiply {
   // Member functions
   public:
      NormalizeScale(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections);
      virtual ~NormalizeScale();

      virtual void ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag);

      virtual int normalizeWeights();

   protected:
      NormalizeScale();
      int initialize(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections);
      virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   private:
      int initialize_base();

   };

   } /* namespace PV */
#endif /* NORMALIZESCALE_HPP_ */
