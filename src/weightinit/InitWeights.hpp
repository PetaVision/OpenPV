/*
 * InitWeights.hpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#ifndef INITWEIGHTS_HPP_
#define INITWEIGHTS_HPP_

#include <columns/BaseObject.hpp>
#include <include/pv_common.h>
#include <include/pv_types.h>
#include <io/PVParams.hpp>
#include <connections/HyPerConn.hpp>
#include <weightinit/InitWeightsParams.hpp>

namespace PV {

//class HyPerCol;
//lass HyPerLayer;
class InitWeightsParams;
//class InitGauss2DWeightsParams;

class InitWeights : public BaseObject {
public:
   InitWeights(char const * name, HyPerCol * hc);
   virtual ~InitWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int communicateParamsInfo();

   /*
    * initializeWeights is not virtual.  It checks initFromLastFlag and then
    * filename, loading weights from a file if appropriate.  Otherwise
    * it calls calcWeights with no arguments.
    * For most InitWeights objects, calcWeights(void) does not have to be
    * overridden but calcWeights(dataStart, patchIndex, arborId) should be.
    * For a few InitWeights classes (e.g. InitDistributedWeights),
    * calcWeights(void) is overridden: a fixed number of weights is active,
    * so it is more convenient and efficient to handle all the weights
    * together than to call one patch at a time.
    */
   int initializeWeights(PVPatch *** patches, pvwdata_t ** dataStart,
			double * timef = NULL);
   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights();
   virtual int calcWeights(pvwdata_t * dataStart,
			int patchIndex, int arborId);

   virtual int readWeights(PVPatch *** patches, pvwdata_t ** dataStart, int numPatches,
                           const char * filename, double * time=NULL);

protected:
   InitWeights();
   int initialize(const char * name, HyPerCol * hc);
   virtual int initRNGs(bool isKernel) { return PV_SUCCESS; }
   virtual int zeroWeightsOutsideShrunkenPatch(PVPatch *** patches);
   virtual int readListOfArborFiles(PVPatch *** patches, pvwdata_t ** dataStart,int numPatches,
         const char * listOfArborsFilename, double * timef=NULL);
   virtual int readCombinedWeightFiles(PVPatch *** patches, pvwdata_t ** dataStart,int numPatches,
         const char * fileOfWeightFiles, double * timef=NULL);

private:
   int initialize_base();

protected:
   HyPerConn * callingConn;
   InitWeightsParams * weightParams;

}; // class InitWeights

BaseObject * createInitWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITWEIGHTS_HPP_ */
