/*
 * UniformRandomV.hpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#ifndef UNIFORMRANDOMV_HPP_
#define UNIFORMRANDOMV_HPP_

#include "BaseInitV.hpp"

namespace PV {

class UniformRandomV : public BaseInitV {
  protected:
   /**
    * List of parameters needed from the UniformRandomV class
    * @name UniformRandomV Parameters
    * @{
    */

   /**
    * @brief minV: The minimum value of the random distribution
    */
   virtual void ioParam_minV(ParamsIOSwitch ioSwitch);

   /**
    * @brief maxV: The maximum value of the random distribution
    */
   virtual void ioParam_maxV(ParamsIOSwitch ioSwitch);
   /** @} */
  public:
   UniformRandomV(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~UniformRandomV();
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void calcV(float *V, PVLayerLoc const *loc) override;

  protected:
   UniformRandomV();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  private:
   int initialize_base();

   // data members
  private:
   float mMinV = (float)0;
   float mMaxV = (float)1;

}; // end class UniformRandomV

} // end namespace PV

#endif /* UNIFORMRANDOMV_HPP_ */
