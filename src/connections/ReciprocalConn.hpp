/*
 * ReciprocalConn.hpp
 *
 *  Created on: Feb 16, 2012
 *      Author: pschultz
 */

#ifndef RECIPROCALCONN_HPP_
#define RECIPROCALCONN_HPP_

#include "KernelConn.hpp"
#include "../utils/pv_random.h"

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
   const char * getReciprocalWgtsName()       {return reciprocalWgtsName;}
   float getReciprocalFidelityCoeff()         {return reciprocalFidelityCoeff;}
   bool getSlownessFlag()                     {return slownessFlag;}
   int getSizeUnitCellPost()                  {return sizeUnitCellPost;}

   int setReciprocalWgts(const char * recipName);

   virtual int updateState(float timef, float dt);

protected:
   // protected methods
   ReciprocalConn();
   int initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post,
         ChannelType channel, const char * filename,
         InitWeights *weightInit=NULL);
   int initParameterLayer(const char * parametername, HyPerLayer ** layerPtr,
         HyPerLayer * defaultlayer=NULL);
   virtual int initNormalize();
   virtual int update_dW(int axonID);
   virtual int updateWeights(int axonId);
   virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborID);

   pvdata_t * getSums()         {return sums;}

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
   float relaxationRate; // The coefficient eta in dW = eta * dE/dW, measured in the same units as HyPerCol's dt
   float reciprocalFidelityCoeff;
   bool slownessFlag;
   HyPerLayer * slownessPre;
   HyPerLayer * slownessPost;
   int nxUnitCellPost;
   int nyUnitCellPost;
   int nfUnitCellPost;
   int sizeUnitCellPost;
   pvdata_t * sums; // Used in normalizeWeights
   pvdata_t normalizeNoiseLevel; // Used in normalizeWeights
};

} /* namespace PV */
#endif /* RECIPROCALCONN_HPP_ */
