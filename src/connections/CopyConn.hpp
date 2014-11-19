/*
 * CopyConn.hpp
 *
 *  Created on: Nov 19, 2014
 *      Author: pschultz
 */

#ifndef COPYCONN_HPP_
#define COPYCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class CopyConn: public HyPerConn {
public:
   CopyConn(char const * name, HyPerCol * hc);
   virtual ~CopyConn();
   virtual int communicateInitInfo();
   virtual bool needUpdate(double time, double dt);

   char const * getOriginalConnName() { return originalConnName; }
   HyPerConn * getOriginalConn() { return originalConn; }

protected:
   CopyConn();
   int initialize(char const * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   /**
    * List of parameters needed from the CopyConn class
    * @name CopyConn Parameters
    * @{
    */

   /**
    * @brief weightInitType is not used by CopyConn.
    */
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);

   /**
    * @brief CopyConn inherits numAxonalArbors the original connection, instead of reading it from parameters
    */
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);

   /**
    * @brief CopyConn inherits plasticityFlag the original connection, instead of reading it from parameters
    */
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief CopyConn inherits triggerFlag the original connection, instead of reading it from parameters
    */
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief weightUpdatePeriod is not used by CopyConn.
    */
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);

   /**
     * @brief initialWeightUpdateTime is not used by CopyConn.
     */
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief originalConnName (required): The name of the connection the weights will be copied from
    */
   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
   /** @} */

   virtual int updateWeights(int arborId = 0);
   int copy(int arborId = 0);

   char * originalConnName;
   HyPerConn * originalConn;

private:
   int initialize_base();
};

} /* namespace PV */

#endif /* COPYCONN_HPP_ */
