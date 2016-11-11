/*
 * BaseInitV.hpp
 *
 *  Created on: Oct 25, 2016
 *      Author: pschultz
 */

#ifndef BASEINITV_HPP_
#define BASEINITV_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

class BaseInitV : public BaseObject {
  public:
   BaseInitV(char const *name, HyPerCol *hc);
   virtual ~BaseInitV();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int calcV(float *V, PVLayerLoc const *loc);

  protected:
   BaseInitV();
   int initialize(char const *name, HyPerCol *hc);

  private:
   int initialize_base();
}; // end class BaseInitV

} // end namespace PV

#endif /* BASEINITV_HPP_ */
