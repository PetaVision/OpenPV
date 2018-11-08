/*
 * ConstantV.hpp
 *
 *  Created on: Oct 26, 2011
 *      Author: pschultz
 */

#ifndef ZEROV_HPP_
#define ZEROV_HPP_

#include "ConstantV.hpp"

namespace PV {

class ZeroV : public ConstantV {
  protected:
   /**
    * List of parameters needed for the ConstantV class
    * @name ConstantV Parameters
    * @{
    */

   /**
    * @brief valueV: ZeroV does not read valueV but sets it to zero.
    */
   virtual void ioParam_valueV(enum ParamsIOFlag ioFlag) override;
   /** @} */

  public:
   ZeroV(const char *name, PVParams *params, Communicator *comm);
   virtual ~ZeroV();

  protected:
   ZeroV();
   void initialize(const char *name, PVParams *params, Communicator *comm);

  private:
   int initialize_base();
}; // end class ZeroV

} // end namespace PV

#endif /* ZEROV_HPP_ */
