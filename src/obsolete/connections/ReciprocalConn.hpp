/*
 * ReciprocalConn.hpp
 *
 *  Created on: Feb 16, 2012
 *      Author: pschultz
 */

#ifndef RECIPROCALCONN_HPP_
#define RECIPROCALCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class ReciprocalConn: public PV::HyPerConn {
public:
   // public methods
   ReciprocalConn(const char * name, HyPerCol * hc);
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

   int setReciprocalWgts(const char * recipName);

protected:
   // protected methods
   ReciprocalConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_relaxationRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_reciprocalFidelityCoeff(enum ParamsIOFlag ioFlag);
   virtual void ioParam_updateRulePre(enum ParamsIOFlag ioFlag);
   virtual void ioParam_updateRulePost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_slownessFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_slownessPre(enum ParamsIOFlag ioFlag);
   virtual void ioParam_slownessPost(enum ParamsIOFlag ioFlag);
   virtual void ioParam_reciprocalWgts(enum ParamsIOFlag ioFlag);
   int getLayerName(PVParams * params, const char * parameter_name, char ** layer_name_ptr, const char * default_name=NULL);
   int setParameterLayer(const char * paramname, const char * layername, HyPerLayer ** layerPtr);
   virtual int update_dW(int axonID);
   virtual int updateWeights(int axonId);
   int getReciprocalWgtCoordinates(int kx, int ky, int kf, int kernelidx, int * kxRecip, int * kyRecip, int * kfRecip, int * kernelidxRecip);

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
   char * reciprocalWgtsName;
   ReciprocalConn * reciprocalWgts;
   float relaxationRate; // The coefficient eta in dW = eta * dE/dW, measured in the same units as HyPerCol's dt
   float reciprocalFidelityCoeff;
   bool slownessFlag;
   char * slownessPreName;
   char * slownessPostName;
   HyPerLayer * slownessPre;
   HyPerLayer * slownessPost;
};

} /* namespace PV */
#endif /* RECIPROCALCONN_HPP_ */
