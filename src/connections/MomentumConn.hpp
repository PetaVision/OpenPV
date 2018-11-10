/* MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef MOMENTUMCONN_HPP_
#define MOMENTUMCONN_HPP_

#include "connections/HyPerConn.hpp"

namespace PV {


class MomentumConn : public HyPerConn {
  public:
   MomentumConn(char const *name, PVParams *params, Communicator *comm);

   virtual ~MomentumConn();

  protected:
   MomentumConn();

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual BaseWeightUpdater *createWeightUpdater() override;
}; // class MomentumConn

} // namespace PV

#endif // MOMENTUMCONN_HPP_
