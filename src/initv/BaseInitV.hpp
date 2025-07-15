/*
 * BaseInitV.hpp
 *
 *  Created on: Oct 25, 2016
 *      Author: pschultz
 */

#ifndef BASEINITV_HPP_
#define BASEINITV_HPP_

#include "columns/BaseObject.hpp"
#include <string>

namespace PV {

class BaseInitV : public BaseObject {
  public:
   BaseInitV(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~BaseInitV();
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void calcV(float *V, PVLayerLoc const *loc);

  protected:
   BaseInitV();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual void setObjectType() override;

  private:
   int initialize_base();

}; // end class BaseInitV

} // end namespace PV

#endif /* BASEINITV_HPP_ */
