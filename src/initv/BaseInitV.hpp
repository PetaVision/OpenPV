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
   BaseInitV(char const *name, PVParams *params, Communicator *comm);
   virtual ~BaseInitV();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void calcV(float *V, PVLayerLoc const *loc);

  protected:
   BaseInitV();
   void initialize(char const *name, PVParams *params, Communicator *comm);
   virtual void setObjectType() override;

  private:
   int initialize_base();

  public:
   static string const mDefaultInitV;
}; // end class BaseInitV

} // end namespace PV

#endif /* BASEINITV_HPP_ */
