/*
 * InitV.hpp
 *
 *  Created on: Dec 6, 2011
 *      Author: pschultz
 */

#ifndef INITV_HPP_
#define INITV_HPP_

#include "../include/default_params.h"
#include "../include/pv_types.h"
#include "../include/pv_common.h"
#include "../utils/pv_random.h"
#include "../layers/HyPerLayer.hpp"
#include "../io/fileio.hpp"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include <stdarg.h>

namespace PV {

enum InitVType {
   UndefinedInitV,
   ConstantV,
   UniformRandomV,
   GaussianRandomV,
   InitVFromFile
};

class InitV {

public:
   InitV(HyPerCol * hc, const char * groupName);
   virtual ~InitV();
   virtual int calcV(HyPerLayer * layer);

protected:
   InitV();
   int initialize(HyPerCol * hc, const char * groupName);

private:
   int initialize_base();
   int calcConstantV(pvdata_t * V, int numNeurons);
   int calcGaussianRandomV(pvdata_t * V, int numNeurons);
   pvdata_t generateGaussianRand();
   int calcUniformRandomV(pvdata_t * V, int numNeurons);
   pvdata_t generateUnifRand();
   int calcVFromFile(PVLayer * clayer, InterColComm * icComm);
   int checkLoc(const PVLayerLoc * loc, int nx, int ny, int nf, int nxGlobal, int nyGlobal);
   int checkLocValue(int fromParams, int fromFile, const char * field);
   int printerr(const char * fmtstring, ...);

   char * groupName;
   InitVType initVTypeCode;
   pvdata_t constantValue; // Defined only for initVTypeCode=ConstantV
   pvdata_t minV, maxV, uniformMultiplier; // Defined only for initVTypeCode=UniformRandomV
      // uniformMultiplier converts the primary random number to a double between 0 and maxV-minV
   pvdata_t meanV, sigmaV, heldValue; bool valueIsBeingHeld;// Defined only for GaussianRandomV
      // if valueIsBeingHeld is true, heldValue is normally distributed random number with mean meanV, st.dev. sigmaV
      // if valueIsBeingHeld is false, heldValue is undefined
   const char * filename; // Defined only for initVTypeCode=InitVFromFile
   bool useStderr; // If true, printerr passes message to stderr.  If false, printerr does nothing.

}; // end class InitV

}  // end namespace PV


#endif /* INITV_HPP_ */
