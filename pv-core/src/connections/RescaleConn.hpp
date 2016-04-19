/*
 * RescaleConn.hpp
 *
 *  Created on: Apr 15, 2016
 *      Author: pschultz
 */

#ifndef PV_CORE_SRC_CONNECTIONS_RESCALECONN_HPP_
#define PV_CORE_SRC_CONNECTIONS_RESCALECONN_HPP_

#include "IdentConn.hpp"

namespace PV {

class RescaleConn: public IdentConn {
public:
   RescaleConn(char const * name, HyPerCol * hc);
   virtual ~RescaleConn();

protected:
   RescaleConn();
   int initialize(char const * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * List of parameters needed from the HyPerConn class
    * @name HyPerConn Parameters
    * @{
    */

   /**
    * scale: presynaptic activity is multiplied by this scale factor before being added to the postsynaptic input.
    */
   void ioParam_scale(enum ParamsIOFlag ioFlag);

   /** @} */
   // End of parameters needed from the RescaleConn class.

   virtual int deliverPresynapticPerspective(PVLayerCube const * activity, int arborID);

private:
   int initialize_base();

// Member variables
protected:
   float scale;
};

BaseObject * createRescaleConn(char const * name, HyPerCol * hc);

} /* namespace PV */

#endif /* PV_CORE_SRC_CONNECTIONS_RESCALECONN_HPP_ */
