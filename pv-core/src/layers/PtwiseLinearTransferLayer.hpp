/*
 * PtwiseLinearTransferLayer.hpp
 *
 *  Created on: July 24, 2015
 *      Author: pschultz
 */

#ifndef PTWISELINEARTRANSFERLAYER_HPP_
#define PTWISELINEARTRANSFERLAYER_HPP_

#include "HyPerLayer.hpp"
#include "../include/pv_datatypes.h"

namespace PV {

class PtwiseLinearTransferLayer : public HyPerLayer {
public:

   /**
    * PtwiseLinearTransferLayer is a layer where the activity is
    * related to the membrane potential through a piece-wise linear transfer
    * function.  The transfer function can have jumps, in which case it
    * is continuous from the right.  The transfer function is defined
    * by specifying the V- and A-coordinates of the vertices of the graph,
    * and the slopes going to positive and negative infinity.
    * (See the parameters verticesV, verticesA, slopeNegInf, and slopePosInf.)
    */
   PtwiseLinearTransferLayer(const char* name, HyPerCol * hc);

   virtual ~PtwiseLinearTransferLayer();

   /**
    * Returns the number of points in verticesV and verticesA.
    */
   int getNumVertices()         { return numVertices; }

   /**
    * Returns the V-coordinate of the the nth vertex (zero-indexed).
    * If n is out of bounds, returns NaN.
    */
   pvdata_t getVertexV(int n)   { if (n>=0 && n<numVertices) { return verticesV[n]; } else { return nan(""); } }

   /**
    * Returns the V-coordinate of the the nth vertex (zero-indexed).
    * If n is out of bounds, returns NaN.
    */
   pvdata_t getVertexA(int n)   { if (n>=0 && n<numVertices) { return verticesA[n]; } else { return nan(""); } }
   pvdata_t getSlopeNegInf()    { return slopeNegInf; }
   pvdata_t getSlopePosInf()    { return slopePosInf; }

   virtual bool activityIsSpiking() { return false; }
protected:
   PtwiseLinearTransferLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);
   virtual int setActivity();
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   
   // To allow doxygen to document the layer's parameters, put all ioParam_<parametername> functions between
   // this comment block and the comment "/** @} */" below.
   /** 
    * List of parameters used by the PtwiseLinearTransferLayer class
    * @name PtwiseLinearTransferLayer Parameters
    * @{
    */

   /**
    * @brief verticesV: An array of membrane potentials at points where the transfer function jumps or changes slope.
    * There must be the same number of elements in verticesV as verticesA, and the sequence of values must
    * be nondecreasing.
    */
   virtual void ioParam_verticesV(enum ParamsIOFlag ioFlag);

   /**
    * @brief verticesA: An array of activities of points where the transfer function jumps or changes slope.
    * There must be the same number of elements in verticesA as verticesV.
    */
   virtual void ioParam_verticesA(enum ParamsIOFlag ioFlag);

   /**
    * @brief slopeNegInf: The slope of the transfer function when x is less than the first element of verticesV.
    * Thus, if V < Vfirst, the corresponding value of A is A = Afirst - slopeNegInf * (Vfirst - V)
    */
   virtual void ioParam_slopeNegInf(enum ParamsIOFlag ioFlag);

   /**
    * @brief slopePosInf: The slope of the transfer function when x is greater than the last element of verticesV.
    * Thus, if V > Vlast, the corresponding value of A is A = Alast + slopePosInf * (V - Vlast)
    */
   virtual void ioParam_slopePosInf(enum ParamsIOFlag ioFlag);

   /**
    * @brief clearGSynInterval: the time interval after which GSyn is reset to zero.
    * @details Until this interval elapses, GSyn continues to accumulate from timestep to timestep.
    * If clearGSynInterval is zero or negative, GSyn clears every timestep.
    * If clearGSynInterval is infinite, the layer acts as an accumulator.
    */
   virtual void ioParam_clearGSynInterval(enum ParamsIOFlag ioFlag);
   /** @} */

   virtual int checkVertices();
   int setSlopes();

   virtual int resetGSynBuffers(double timef, double dt);

   virtual int checkpointRead(const char * cpDir, double * timeptr); // (const char * cpDir, double * timed);
   virtual int checkpointWrite(const char * cpDir);

//#ifdef PV_USE_OPENCL
//   virtual int getNumCLEvents() {return numEvents;}
//   virtual const char * getKernelName() { return "PtwiseLinearTransferLayer_update_state"; }
//   virtual int initializeThreadBuffers(const char * kernel_name);
//   virtual int initializeThreadKernels(const char * kernel_name);
//   //virtual int getEVActivity() {return EV_ANN_ACTIVITY;}
//   int updateStateOpenCL(double time, double dt);
//   //temporary method for debuging recievesynapticinput
//public:
////   virtual void copyChannelExcFromDevice() {
////      getChannelCLBuffer(CHANNEL_EXC)->copyFromDevice(&evList[getEVGSynE()]);
////      clWaitForEvents(1, &evList[getEVGSynE()]);
////      clReleaseEvent(evList[getEVGSynE()]);
////   }
//protected:
//#endif // PV_USE_OPENCL

private:
   int initialize_base();

// Member variables
protected:
   int numVertices;
   pvpotentialdata_t * verticesV;
   pvadata_t * verticesA;
   float * slopes; // slopes[0]=slopeNegInf; slopes[numVertices]=slopePosInf; slopes[k]=slope from vertex k-1 to vertex k
   float slopeNegInf;
   float slopePosInf;

   double clearGSynInterval; // The interval between successive clears of GSyn
   double nextGSynClearTime; // The next time that the GSyn will be cleared.
}; // end of class PtwiseLinearTransferLayer

BaseObject * createPtwiseLinearTransferLayer(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* PTWISELINEARTRANSFERLAYER_HPP_ */
