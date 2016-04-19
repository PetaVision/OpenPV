/*
 * GradientCheckConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef GRADIENTCHECKCONN_HPP_
#define GRADIENTCHECKCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PVMLearning {

class GradientCheckConn: public PV::HyPerConn{
public:
   GradientCheckConn(const char * name, PV::HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateState(double time, double dt);
protected:
   GradientCheckConn();
   int initialize_base();
   float getSqErrCost();
   float getLogErrCost();
   float getCost();
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_gtLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_estLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_costFunction(enum ParamsIOFlag ioFlag);
   int calc_dW() ;
   //virtual int clear_dW();

private:
   char* estLayerName;
   char* gtLayerName;
   char* costFunction;
   PV::HyPerLayer* estLayer;
   PV::HyPerLayer* gtLayer;
   bool firstRun;
   bool secondRun;
   float origCost;
   float currCost;
   float epsilon;
   long prevIdx;
   float prevWeightVal;
}; // end of class GradientCheckConn

PV::BaseObject * createGradientCheckConn(char const * name, PV::HyPerCol * hc);

}  // end of block for namespace PVMLearning


#endif /* GRADIENTCHECKCONN_HPP_ */
