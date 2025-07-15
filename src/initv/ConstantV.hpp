/*
 * ConstantV.hpp
 *
 *  Created on: Oct 26, 2011
 *      Author: pschultz
 */

#ifndef CONSTANTV_HPP_
#define CONSTANTV_HPP_

#include "BaseInitV.hpp"

namespace PV {

class ConstantV : public BaseInitV {
  protected:
   /**
    * List of parameters needed for the ConstantV class
    * @name ConstantV Parameters
    * @{
    */

   /**
    * @brief valueV: The value to initialize the V buffer with
    */
   virtual void ioParam_valueV(ParamsIOSwitch ioSwitch);
   /** @} */

  public:
   ConstantV(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~ConstantV();
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void calcV(float *V, PVLayerLoc const *loc) override;

  protected:
   ConstantV();
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  private:
   int initialize_base();

   // Data members
  protected:
   float mValueV;
}; // end class ConstantV

} // end namespace PV

#endif /* CONSTANTV_HPP_ */
