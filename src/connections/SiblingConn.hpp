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
   SiblingConn(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename = NULL, InitWeights *weightInit = NULL,
         const char * sibling_conn_name=NULL);
   virtual int communicateInitInfo();
   bool getIsNormalized();
   void setSiblingConn(SiblingConn *sibling_conn);
   SiblingConn * getSiblingConn(){return siblingConn;};

protected:
   int initialize_base(){return PV_SUCCESS;};
   int initialize(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename, InitWeights *weightInit=NULL,
         const char * sibling_conn_name=NULL);
   virtual int initNormalize();
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
