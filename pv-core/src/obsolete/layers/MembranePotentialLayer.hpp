/*
 * MembranePotentialLayer.hpp
 *
 *  Created on: Mar 20, 2014
 *      Author: pschultz
 */

#ifndef MEMBRANEPOTENTIALLAYER_HPP_
#define MEMBRANEPOTENTIALLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class MembranePotentialLayer: public PV::HyPerLayer {
public:
   MembranePotentialLayer(const char * name, HyPerCol * hc);
   virtual ~MembranePotentialLayer();

protected:
   MembranePotentialLayer();
   int initialize(const char * name, HyPerCol * hc);

private:
   int initialize_base();

};

} /* namespace PV */
#endif /* MEMBRANEPOTENTIALLAYER_HPP_ */
