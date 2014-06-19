/*
 * PlasticCloneConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef PLASTICCLONECONN_HPP_
#define PLASTICCLONECONN_HPP_

#include "CloneConn.hpp"

namespace PV {

class PlasticCloneConn : public CloneConn{

public:
   PlasticCloneConn(const char * name, HyPerCol * hc);

   virtual int communicateInitInfo();

protected:
   PlasticCloneConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
   virtual void ioParam_useWindowPost(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

}; // end class PlasticCloneConn

}  // end namespace PV

#endif /* CLONECONN_HPP_ */
