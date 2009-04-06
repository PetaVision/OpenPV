/*
 * V1_S1.hpp
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

#ifndef V1_S1_HPP_
#define V1_S1_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class V1_S1: public PV::HyPerLayer {
public:
   V1_S1(const char * name, HyPerCol * hc);

   virtual int columnWillAddLayer(InterColComm * comm, int layerId);
   virtual int updateState(float time, float dt);

protected:
   int no;
   pvdata_t * subWeights;
};

} // namespace PV

#endif /* V1_S1_HPP_ */
