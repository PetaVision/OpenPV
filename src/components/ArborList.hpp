/*
 * ArborList.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef ARBORLIST_HPP_
#define ARBORLIST_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the numAxonalArbors parameter and the delay parameters array.
 */
class ArborList : public BaseObject {
  protected:
   /**
    * List of parameters needed from the ArborList class
    * @name ArborList Parameters
    * @{
    */

   /**
    * @brief numAxonalArbors: Specifies the number of arbors to use in the connection
    */
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);

   /**
    * @brief delay: Specifies delay(s) which the post layer will receive data
    * @details: Delays are specified in units of dt, but are rounded to be integer multiples of dt.
    * If delay is a scalar, all arbors of the connection have that value of delay.
    * If delay is an array, the length must match the number of arbors and the arbors are assigned
    * the delays sequentially.
    * If the delay parameter is omitted, all delays will be zero.
    */
   virtual void ioParam_delay(enum ParamsIOFlag ioFlag);

   /** @} */ // end of ArborList parameters

  public:
   ArborList(char const *name, HyPerCol *hc);
   virtual ~ArborList();

   virtual void setObjectType() override;

   /**
    * Returns the number of arbors in the connection
    */
   int getNumAxonalArbors() const { return mNumAxonalArbors; }

   int getDelay(int arbor) const { return mDelay[arbor]; }

  protected:
   ArborList();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void initializeDelays();

   void setDelay(int arborId, double delay);

   int maxDelaySteps();

  protected:
   int mNumAxonalArbors = 1;
   std::vector<int> mDelay; // The delays expressed in # of timesteps (delays ~= fDelayArray / t)
   double *mDelaysParams = nullptr; // The raw delays in params, in the same units that dt is in.
   int mNumDelays        = 0; // The size of the mDelayParams array

}; // class ArborList

} // namespace PV

#endif // ARBORLIST_HPP_
