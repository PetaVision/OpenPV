/*
 * BaseLayer.hpp
 *
 *  Created on: Jan 16, 2010
 *      Author: rasmussn
 */

#ifndef BASELAYER_HPP_
#define BASELAYER_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * The base class for layers.  BaseLayer should not be instantiated itself;
 * instead, instantiate classes derived from BaseLayer.
 *
 * BaseLayer should not be templated; if the occasion arises to template
 * classes, this should be done at the HyPerLayer level or below.
 * The rationale is that HyPerCol stores an array of layers, and needs
 * a class that all layers, however templated, are derived from.
 */
class BaseLayer : public BaseObject {
public:
   virtual ~BaseLayer();

protected:
   BaseLayer();
   int initialize(char const * name, HyPerCol * hc);
};

} // namespace PV

#endif /* BASELAYER_HPP_ */
