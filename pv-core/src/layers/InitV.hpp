/*
 * InitV.hpp
 *
 *  Created on: Dec 6, 2011
 *      Author: pschultz
 */

#ifndef INITV_HPP_
#define INITV_HPP_

#include "../columns/Random.hpp"
#include "../columns/GaussianRandom.hpp"
#include "../include/default_params.h"
#include "../include/pv_types.h"
#include "../include/pv_common.h"
#include "../layers/HyPerLayer.hpp"
#include "../io/fileio.hpp"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include <stdarg.h>
//#include "Image.hpp"

namespace PV {

enum InitVType {
   UndefinedInitV,
   ConstantV,
   UniformRandomV,
   GaussianRandomV,
   InitVFromFile
};

class InitV {
protected:
   /** 
    * List of parameters needed from the InitV class
    * @name InitV Parameters
    * @{
    */
   
   /**
    * @brief valueV: The value to initialize the V buffer with
    */
   virtual void ioParamGroup_ConstantV(enum ParamsIOFlag ioFlag);

   /**
    * @brief No other parameters nessessary
    */
   virtual void ioParamGroup_ZeroV(enum ParamsIOFlag ioFlag);

   /**
    * @brief minV: The minimum value to generate uniform random V's with <br />
    * @brief maxV: The maximum value to generate uniform random V's with
    */
   virtual void ioParamGroup_UniformRandomV(enum ParamsIOFlag ioFlag);

   /**
    * @brief meanV: The mean to generate guassian random V's with <br />
    * @brief sigmaV: The standard deviation to generate gaussian random V's with
    */
   virtual void ioParamGroup_GaussianRandomV(enum ParamsIOFlag ioFlag);

   /**
    * @brief Vfilename: The pvp filename to load the V buffer from
    */
   virtual void ioParamGroup_InitVFromFile(enum ParamsIOFlag ioFlag);
   /** @} */
public:
   InitV(HyPerCol * hc, const char * groupName);
   virtual ~InitV();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int calcV(HyPerLayer * layer);

protected:
   InitV();
   int initialize(HyPerCol * hc, const char * groupName);


private:
   int initialize_base();
   int calcConstantV(pvdata_t * V, int numNeurons);
   int calcGaussianRandomV(pvdata_t * V, const PVLayerLoc * loc, HyPerCol * hc);
   int calcUniformRandomV(pvdata_t * V, const PVLayerLoc * loc, HyPerCol * hc);
   int calcVFromFile(pvdata_t * V, const PVLayerLoc * loc, InterColComm * icComm);
   int checkLoc(const PVLayerLoc * loc, int nx, int ny, int nf, int nxGlobal, int nyGlobal);
   int checkLocValue(int fromParams, int fromFile, const char * field);
   int printerr(const char * fmtstring, ...);

   HyPerCol * parent;
   char * groupName;
   char * initVTypeString;
   InitVType initVTypeCode;
   pvdata_t constantValue; // Defined only for initVTypeCode=ConstantV
   pvdata_t minV, maxV; // Defined only for initVTypeCode=UniformRandomV
      // uniformMultiplier converts the primary random number to a double between 0 and maxV-minV
   pvdata_t meanV, sigmaV; // Defined only for initVTypeCode=GaussianRandomV
      // if valueIsBeingHeld is true, heldValue is normally distributed random number with mean meanV, st.dev. sigmaV
      // if valueIsBeingHeld is false, heldValue is undefined
   char * filename; // Defined only for initVTypeCode=InitVFromFile
   bool useStderr; // If true, printerr passes message to stderr.  If false, printerr does nothing.
}; // end class InitV

}  // end namespace PV


#endif /* INITV_HPP_ */
