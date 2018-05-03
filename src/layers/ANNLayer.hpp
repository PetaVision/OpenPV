/*
 * ANNLayer.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef ANNLAYER_HPP_
#define ANNLAYER_HPP_

#include "HyPerLayer.hpp"
#include <limits>

#define NUM_ANN_EVENTS 3
#define EV_ANN_ACTIVITY 2

namespace PV {

class ANNLayer : public HyPerLayer {
  public:
   ANNLayer(const char *name, HyPerCol *hc);
   virtual ~ANNLayer();

   /**
    * Returns true if the params file specifies the transfer function using
    * verticesV, verticesA, slopeNegInf, and slopePosInf.
    * Returns false otherwise, in which case the above parameters are
    * computed internally from VThresh, AMin, AMax, AShift, and VWidth.
    */
   bool layerListsVerticesInParams() const { return verticesListInParams; }

   float getVThresh() const { return VThresh; }
   float getAMax() const { return AMax; }
   float getAMin() const { return AMin; }
   float getAShift() const { return AShift; }
   float getVWidth() const { return VWidth; }

   /**
    * Returns the number of points in verticesV and verticesA.
    */
   int getNumVertices() const { return numVertices; }

   /**
    * Returns the V-coordinate of the the nth vertex (zero-indexed).
    * If n is out of bounds, returns NaN.
    */
   float getVertexV(int n) const {
      if (n >= 0 && n < numVertices) {
         return verticesV[n];
      }
      else {
         return nan("");
      }
   }

   /**
    * Returns the V-coordinate of the the nth vertex (zero-indexed).
    * If n is out of bounds, returns NaN.
    */
   float getVertexA(int n) const {
      if (n >= 0 && n < numVertices) {
         return verticesA[n];
      }
      else {
         return nan("");
      }
   }
   float getSlopeNegInf() const { return slopeNegInf; }
   float getSlopePosInf() const { return slopePosInf; }

   virtual bool activityIsSpiking() override { return false; }

  protected:
   ANNLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual Response::Status updateState(double time, double dt) override;
   virtual int setActivity() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * List of parameters used by the ANNLayer class
    * @name ANNLayer Parameters
    * @{
    */

   /**
    * @brief verticesV: An array of membrane potentials at points where the transfer function jumps
    * or changes slope.
    * There must be the same number of elements in verticesV as verticesA, and the sequence of
    * values must
    * be nondecreasing.  If this parameter is absent, layerListsVerticesInParams() returns false
    * and the vertices are computed internally from VThresh, AMin, AMax, AShift, and VWidth.
    * If the parameter is present, layerListsVerticesInParams() returns true.
    */
   virtual void ioParam_verticesV(enum ParamsIOFlag ioFlag);

   /**
    * @brief verticesA: An array of activities of points where the transfer function jumps or
    * changes slope.
    * There must be the same number of elements in verticesA as verticesV.
    * Only read if verticesV is present; otherwise it is computed internally from VThresh, AMin,
    * AMax, AShift, and VWidth.
    */
   virtual void ioParam_verticesA(enum ParamsIOFlag ioFlag);

   /**
    * @brief slopeNegInf: The slope of the transfer function when x is less than the first element
    * of verticesV.
    * Thus, if V < Vfirst, the corresponding value of A is A = Afirst - slopeNegInf * (Vfirst - V)
    * Only read if verticesV is present; otherwise it is computed internally from VThresh, AMin,
    * AMax, AShift, and VWidth.
    */
   virtual void ioParam_slopeNegInf(enum ParamsIOFlag ioFlag);

   /**
    * @brief slopePosInf: The slope of the transfer function when x is greater than the last element
    * of verticesV.
    * Thus, if V > Vlast, the corresponding value of A is A = Alast + slopePosInf * (V - Vlast)
    * Only read if verticesV is present; otherwise it is computed internally from VThresh, AMin,
    * AMax, AShift, and VWidth.
    */
   virtual void ioParam_slopePosInf(enum ParamsIOFlag ioFlag);

   /**
    * @brief VThresh: Only read if verticesV is absent.
    * The threshold value for the membrane potential.  Below this value, the
    * output activity will be AMin.  Above, it will obey the transfer function
    * as specified by AMax, VWidth, and AShift.  Default is -infinity.
    */
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);

   /**
    * @brief AMin: Only read if verticesV is absent.
    * When membrane potential V is below the threshold VThresh, activity
    * takes the value AMin.  Default is the value of VThresh.
    */
   virtual void ioParam_AMin(enum ParamsIOFlag ioFlag);

   /**
    * @brief AMax: Only read if verticesV is absent.
    * Activity that would otherwise be greater than AMax is truncated to AMax.
    * Default is +infinity.
    */
   virtual void ioParam_AMax(enum ParamsIOFlag ioFlag);

   /**
    * @brief AShift: Only read if verticesV is absent.
    * When membrane potential V is above the threshold VThresh, activity is V-AShift
    * (but see VWidth for making a gradual transition at VThresh).  Default is zero.
    */
   virtual void ioParam_AShift(enum ParamsIOFlag ioFlag);

   /**
    * @brief VWidth: Only read if verticesV is absent.
    * When the membrane potential is between VThresh and VThresh+VWidth, the activity changes
    * linearly
    * between A=AMin when V=VThresh and A=VThresh+VWidth-AShift when V=VThresh+VWidth.
    * Default is zero.
    */
   virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);

   /** @} */

   /**
    * If the params file does not specify verticesV and verticesA explicitly,
    * ANNLayer::initialize() calls this function to compute the vertices.
    */
   virtual int setVertices();

   /**
    * ANNLayer::initialize() calls this function to perform sanity checking
    * on the vertices.  Returns PV_SUCCESS or PV_FAILURE to indicate whether
    * the vertices are acceptable.
    * For ANNLayer::checkVertices(), fails if the sequence of V vertices ever
    * decreases.  If the sequence of A vertices ever decreases, outputs a warning
    * but does not fail.
    */
   virtual int checkVertices() const;

   /**
    * ANNLayer::initialize() calls this function to compute the slopes between
    * vertices.
    */
   void setSlopes();

   virtual int resetGSynBuffers(double timef, double dt) override;

   // Data members, initialized to default values.
   bool verticesListInParams =
         false; // True if verticesV/verticesA were specified in params explicitly; false otherwise
   int numVertices  = 0;
   float *verticesV = nullptr;
   float *verticesA = nullptr;
   float *slopes    = nullptr; // slopes[0]=slopeNegInf; slopes[numVertices]=slopePosInf;
   // slopes[k]=slope from vertex k-1 to vertex k
   float slopeNegInf = 1.0f;
   float slopePosInf = 1.0f;

   float VThresh = -FLT_MAX; // threshold potential, values smaller than VThresh are set to AMin
   float AMax    = FLT_MAX; // maximum membrane potential, larger values are set to AMax
   float AMin    = -FLT_MAX; // minimum membrane potential, smaller values are set to AMin
   float AShift =
         (float)0; // shift potential, values above VThresh are shifted downward by this amount
   // AShift == 0, hard threshold condition
   // AShift == VThresh, soft threshold condition
   float VWidth = (float)0; // The thresholding occurs linearly over the region
   // [VThresh,VThresh+VWidth].  VWidth=0,AShift=0 is standard
   // hard-thresholding

  private:
   int initialize_base();
}; // end of class ANNLayer

} // end namespace PV

#endif /* ANNLAYER_HPP_ */
