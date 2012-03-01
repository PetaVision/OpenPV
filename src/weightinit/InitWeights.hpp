/*
 * InitWeights.hpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#ifndef INITWEIGHTS_HPP_
#define INITWEIGHTS_HPP_

#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../layers/HyPerLayer.hpp"
#include "InitWeightsParams.hpp"
//#include "InitGauss2DWeightsParams.hpp"


namespace PV {

class HyPerCol;
class HyPerLayer;
class InitWeightsParams;
class InitGauss2DWeightsParams;

class InitWeights {
public:
   InitWeights();
   virtual ~InitWeights();

   /*
    * THis is the top weight initializing method.  It will first call createUnShrunkenPatch to create a full sized patch,
    * and then calcWeights to calculate the weights for the patch.  Finally it will call copyToOriginalPatch to copy the weights from
    * the full sized patch to a possibly shrunken patch.
    * This method is purposely not virtual!  Only calcWeights will be virtual and can be over ridden by sub
    * initweights classes
    */
   //void initializeWeights(const char * filename, HyPerConn * callingConn, float * timef=NULL);
   int initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * callingConn, float * timef=NULL);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

   virtual int readWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches,
                           const char * filename, HyPerConn * callingConn, float * time=NULL);

   //get-set methods:
//   inline const char * getName()                     {return name;}


protected:
   virtual int initialize_base();
#ifdef OBSOLETE // Marked obsolete Feb 27, 2012.  With refactoring of wDataStart, there is no need to initialize on an unshrunken patch and copy to a shrunken patch
   PVPatch * createUnShrunkenPatch(HyPerConn * callingConn, PVPatch * wp);
   static int copyToOriginalPatch(PVPatch * wp, PVPatch * wp_tmp, pvdata_t * wtop, int patchIndex, int nf_patch, int sy_patch);
#endif // OBSOLETE
//   char * name; //this is actually the Connection name
//   HyPerLayer     * pre;
//   HyPerLayer     * post;
//   HyPerCol       * parent;
//   HyPerConn      * parentConn;
//   ChannelType channel;    // which channel of the post to update (e.g. inhibit)

private:

   int gauss2DCalcWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitGauss2DWeightsParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITWEIGHTS_HPP_ */
