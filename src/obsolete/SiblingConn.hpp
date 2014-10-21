/*
 * SiblingConn.hpp
 *
 *  Created on: Jan 26, 2012
 *      Author: garkenyon
 */

#ifndef SIBLINGCONN_HPP_
#define SIBLINGCONN_HPP_

#include "NoSelfKernelConn.hpp"

namespace PV {

class SiblingConn: public PV::NoSelfKernelConn {
public:
   SiblingConn(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int communicateInitInfo();
   bool getIsNormalized();
   void setSiblingConn(SiblingConn *sibling_conn);
   SiblingConn * getSiblingConn(){return siblingConn;};


protected:
   int initialize_base(){return PV_SUCCESS;}
   int initialize(const char * name, HyPerCol * hc);
   virtual void ioParam_siblingConnName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
   virtual int normalizeWeights();
   // virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   virtual int normalizeFamily();

private:
   char * siblingConnName;
   SiblingConn * siblingConn;
   bool isNormalized;
};

} /* namespace PV */
#endif /* SIBLINGCONN_HPP_ */
