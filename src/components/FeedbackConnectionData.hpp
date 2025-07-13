/*
 * FeedbackConnectionData.hpp
 *
 *  Created on: Jan 9, 2017
 *      Author: pschultz
 */

#ifndef FEEDBACKCONNECTIONDATA_HPP_
#define FEEDBACKCONNECTIONDATA_HPP_

#include "components/ConnectionData.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

class FeedbackConnectionData : public ConnectionData {
  protected:
   /**
    * List of parameters needed from the FeedbackConnectionData class
    * @name FeedbackConnectionData Parameters
    * @{
    */

   /**
    * @brief preLayerName: FeedbackConnectionData does not read the
    * preLayerName parameter, but takes the pre and post from the
    * original connection and swaps them.
    */
   virtual void ioParam_preLayerName(ParamsIOSwitch ioSwitch) override;

   /**
    * @brief preLayerName: FeedbackConnectionData does not read the
    * preLayerName parameter, but takes the pre and post from the
    * original connection and swaps them.
    */
   virtual void ioParam_postLayerName(ParamsIOSwitch ioSwitch) override;
   /** @} */ // end of FeedbackConnectionData parameters

  public:
   FeedbackConnectionData(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~FeedbackConnectionData();

  protected:
   FeedbackConnectionData();

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

}; // class FeedbackConnectionData

} // namespace PV

#endif // FEEDBACKCONNECTIONDATA_HPP_
