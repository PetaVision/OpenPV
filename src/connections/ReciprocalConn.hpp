/*
 * ReciprocalConn.hpp
 *
 *  Created on: Feb 16, 2012
 *      Author: pschultz
 */

#ifndef RECIPROCALCONN_HPP_
#define RECIPROCALCONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class ReciprocalConn: public PV::KernelConn {
public:
   // public methods
   ReciprocalConn(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post,
         ChannelType channel, const char * filename=NULL,
         InitWeights * weightInit=NULL);
   virtual ~ReciprocalConn();

   const HyPerLayer * getUpdateRulePre()      {return updateRulePre;}
   const HyPerLayer * getUpdateRulePost()     {return updateRulePost;}
   const HyPerLayer * getSlownessPre()        {return slownessPre;}
   const HyPerLayer * getSlownessPost()       {return slownessPost;}
   ReciprocalConn * getReciprocalWgts()       {return reciprocalWgts;}
   float getReciprocalFidelityCoeff()         {return reciprocalFidelityCoeff;}
   bool getSlownessFlag()                     {return slownessFlag;}
   int setReciprocalWgts(const char * recipName);

   virtual int updateState(float time, float dt);

protected:
   // protected methods
   ReciprocalConn();
   int initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post,
         ChannelType channel, const char * filename,
         InitWeights *weightInit=NULL);
   int initParameterLayer(const char * parametername, HyPerLayer ** layerPtr,
         HyPerLayer * defaultlayer=NULL);
   virtual int update_dW(int axonID);
   virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborID);

private:
   // private methods
   int initialize_base();

protected:
   // protected member variables

private:
   // private member variables
   HyPerLayer * updateRulePre;
   HyPerLayer * updateRulePost;
   const char * reciprocalWgtsName;
   ReciprocalConn * reciprocalWgts;
   float reciprocalFidelityCoeff;
   bool slownessFlag;
   HyPerLayer * slownessPre;
   HyPerLayer * slownessPost;
};

} /* namespace PV */
#endif /* RECIPROCALCONN_HPP_ */
