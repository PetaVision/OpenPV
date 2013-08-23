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
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename=NULL, InitWeights * weightInit=NULL);
   virtual ~ReciprocalConn();
   virtual int communicateInitInfo();

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

protected:
   // protected methods
   ReciprocalConn();
   int initialize(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename, InitWeights *weightInit=NULL);
   virtual int setParams(PVParams * params);
   virtual void readRelaxationRate(PVParams * params);
   virtual void readReciprocalFidelityCoeff(PVParams * params);
   virtual int readUpdateRulePre(PVParams * params);
   virtual int readUpdateRulePost(PVParams * params);
   virtual void readSlownessFlag(PVParams * params);
   virtual int readSlownessPre(PVParams * params);
   virtual int readSlownessPost(PVParams * params);
   int getLayerName(PVParams * params, const char * parameter_name, char ** layer_name_ptr, const char * default_name=NULL);
   virtual int readReciprocalWgts(PVParams * params);
   int setParameterLayer(const char * paramname, const char * layername, HyPerLayer ** layerPtr);
   virtual int initNormalize();
   virtual int update_dW(int axonID);
   virtual int updateWeights(int axonId);

private:
   // private methods
   int initialize_base();

protected:
   // protected member variables

private:
   // private member variables
   char * updateRulePreName;
   char * updateRulePostName;
   HyPerLayer * updateRulePre;
   HyPerLayer * updateRulePost;
   const char * reciprocalWgtsName;
   ReciprocalConn * reciprocalWgts;
   float relaxationRate; // The coefficient eta in dW = eta * dE/dW, measured in the same units as HyPerCol's dt
   float reciprocalFidelityCoeff;
   bool slownessFlag;
   char * slownessPreName;
   char * slownessPostName;
   HyPerLayer * slownessPre;
   HyPerLayer * slownessPost;
   int nxUnitCellPost;
   int nyUnitCellPost;
   int nfUnitCellPost;
   int sizeUnitCellPost;
};

} /* namespace PV */
#endif /* RECIPROCALCONN_HPP_ */
