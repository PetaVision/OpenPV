/*
 * DependentArborList.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef DEPENDENTARBORLIST_HPP_
#define DEPENDENTARBORLIST_HPP_

#include "components/ArborList.hpp"

namespace PV {

/**
 * A subclass of ArborList, which retrieves the number of arbors from the connection named
 * in an OriginalConnNameParam component, instead of reading it from params.
 * It still reads the delay array parameter the same way ArborList does.
 */
class DependentArborList : public ArborList {
  protected:
   /**
    * List of parameters needed from the DependentArborList class
    * @name DependentArborList Parameters
    * @{
    */

   /**
    * @brief numAxonalArbors: DependentArborList does not use the numAxonalArbors parameter,
    * but gets the number of arbors from the original connection.
    */
   virtual void ioParam_numAxonalArbors(ParamsIOSwitch ioSwitch) override;

   /** @} */ // end of DependentArborList parameters

  public:
   DependentArborList(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~DependentArborList();

  protected:
   DependentArborList();

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

}; // class DependentArborList

} // namespace PV

#endif // DEPENDENTARBORLIST_HPP_
