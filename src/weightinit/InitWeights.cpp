/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"

#include <stdlib.h>

#include "../include/default_params.h"
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../utils/pv_random.h"
#include "../columns/InterColComm.hpp"
#include "InitGauss2DWeightsParams.hpp"


namespace PV {

InitWeights::InitWeights()
{
   initialize_base();
}

InitWeights::~InitWeights()
{
   // TODO Auto-generated destructor stub
}

/*This method does the three steps involved in initializing weights.  Subclasses usually don't need to override this method.
 * Instead they should override calcWeights to do their own type of weight initialization.
 *
 * This method initializes the full unshrunken patch.  The input argument numPatches is ignored.  Instead, method uses getNumDataPatches to determine number of
 * data patches.
 * For KernelConns, patches should be NULL.
 *
 */
int InitWeights::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart,
		int numPatches, const char * filename, HyPerConn * callingConn,
		double * timef /*default NULL*/) {
	PVParams * inputParams = callingConn->getParent()->parameters();
	int initFromLastFlag = inputParams->value(callingConn->getName(),
			"initFromLastFlag", 0.0f, false) != 0;
	InitWeightsParams *weightParams = NULL;
	int numArbors = callingConn->numberOfAxonalArborLists();
	if (initFromLastFlag) {
		char nametmp[PV_PATH_MAX];
		int chars_needed = snprintf(nametmp, PV_PATH_MAX, "%s/Last/%s_W.pvp",
				callingConn->getParent()->getOutputPath(),
				callingConn->getName());
		if (chars_needed >= PV_PATH_MAX) {
			fprintf(stderr,
					"InitWeights::initializeWeights error: filename \"%s/Last/%s_W.pvp\" is too long.\n",
					callingConn->getParent()->getOutputPath(),
					callingConn->getName());
			abort();
		}
		readWeights(patches, dataStart, callingConn->getNumDataPatches(),
				nametmp, callingConn);
	} else if (filename != NULL) {
		readWeights(patches, dataStart, callingConn->getNumDataPatches(),
				filename, callingConn, timef);
	} else {  // calculate weights
		// modified 2/1/13 by garkenyon: no reason for each process not to calculate weights, which helps in implementing shmget
//		HyPerCol * hc = callingConn->getParent();
//		int root_proc = 0;
//		bool callCalcWeights = patches != NULL || hc->columnId() == root_proc;
//		if (callCalcWeights) {
		weightParams = createNewWeightParams(callingConn);
		for (int arbor = 0; arbor < numArbors; arbor++) {
#ifdef USE_SHMGET
			bool * shmget_owner = callingConn->getShmgetOwnerHead();
			bool shmget_flag = callingConn->getShmgetFlag();
			if (shmget_flag && !shmget_owner[arbor]) continue;
#endif
			for (int dataPatchIndex = 0;
					dataPatchIndex < callingConn->getNumDataPatches();
					dataPatchIndex++) {
				int successFlag = calcWeights(
						callingConn->get_wDataHead(arbor, dataPatchIndex),
						dataPatchIndex, arbor, weightParams);
				if (successFlag != PV_SUCCESS) {
					fprintf(stderr,
							"Failed to create weights for %s! Exiting...\n",
							callingConn->getName());
					exit(PV_FAILURE);
				}
			}
		}
		delete (weightParams);
//		} // callCalcWeights
//		if (patches == NULL) {
//
//			int buf_size = callingConn->xPatchSize() * callingConn->yPatchSize()
//					* callingConn->fPatchSize()
//					* callingConn->getNumDataPatches();
//			MPI_Comm mpi_comm = hc->icCommunicator()->communicator();
//			for (int arbor = 0; arbor < numArbors; arbor++) {
//				MPI_Bcast(callingConn->get_wDataStart(arbor), buf_size,
//						MPI_FLOAT, root_proc, mpi_comm);
//			}
//		}
	}
	return PV_SUCCESS;
}

InitWeightsParams * InitWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitGauss2DWeightsParams(callingConn);
   return tempPtr;
}

int InitWeights::calcWeights(pvdata_t * dataStart, int dataPatchIndex, int arborId,
                               InitWeightsParams *weightParams) {

    InitGauss2DWeightsParams *weightParamPtr = dynamic_cast<InitGauss2DWeightsParams*> (weightParams);

    if(weightParamPtr==NULL) {
       fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
       exit(PV_FAILURE);
    }


    weightParamPtr->calcOtherParams(dataPatchIndex);

    //calculate the weights:
    gauss2DCalcWeights(dataStart, weightParamPtr);


    return PV_SUCCESS;
}

int InitWeights::initialize_base() {

   return PV_SUCCESS;
}

int InitWeights::readWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * conn, double * timef/*default=NULL*/) {
   InterColComm *icComm = conn->getParent()->icCommunicator();
   int numArbors = conn->numberOfAxonalArborLists();
   const PVLayerLoc *preLoc = conn->preSynapticLayer()->getLayerLoc();
   double timed;
   bool useListOfArborFiles = numArbors>1 &&
                              conn->getParent()->parameters()->value(conn->getName(), "useListOfArborFiles", false)!=0;
   bool combineWeightFiles = conn->getParent()->parameters()->value(conn->getName(), "combineWeightFiles", false)!=0;
#ifdef USE_SHMGET
   bool * shmget_owner = conn->getShmgetOwnerHead();
   bool shmget_flag = conn->getShmgetFlag();
#endif
   if( useListOfArborFiles ) {
      int arbor=0;
      FILE * arborfp = pvp_open_read_file(filename, icComm);

      int rootproc = 0;
      char arborfilename[PV_PATH_MAX];
      while( arbor < conn->numberOfAxonalArborLists() ) {
         if( icComm->commRank() == rootproc ) {
            char * fgetsstatus = fgets(arborfilename, PV_PATH_MAX, arborfp);
            if( fgetsstatus == NULL ) {
               bool endoffile = feof(arborfp)!=0;
               if( endoffile ) {
                  fprintf(stderr, "File of arbor files \"%s\" reached end of file before all %d arbors were read.  Exiting.\n", filename, numArbors);
                  exit(EXIT_FAILURE);
               }
               else {
                  int error = ferror(arborfp);
                  assert(error);
                  fprintf(stderr, "File of arbor files: error %d while reading.  Exiting.\n", error);
                  exit(error);
               }
            }
            else {
               // Remove linefeed from end of string
               arborfilename[PV_PATH_MAX-1] = '\0';
               int len = strlen(arborfilename);
               if (len > 1) {
                  if (arborfilename[len-1] == '\n') {
                     arborfilename[len-1] = '\0';
                  }
               }
            }
         } // commRank() == rootproc
         int filetype, datatype;
         int numParams = NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS;
         int params[NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS];
         pvp_read_header(arborfilename, icComm, &timed, &filetype, &datatype, params, &numParams);
         int thisfilearbors = params[INDEX_NBANDS];
#ifndef USE_SHMGET
			int status = PV::readWeights(patches ? &patches[arbor] : NULL, &dataStart[arbor], numArbors-arbor, numPatches, arborfilename, icComm, &timed, preLoc);
#else
			int status = PV::readWeights(patches ? &patches[arbor] : NULL,
					&dataStart[arbor], numArbors - arbor, numPatches,
					arborfilename, icComm, &timed, preLoc, shmget_owner,
					shmget_flag);
#endif
         if (status != PV_SUCCESS) {
            fprintf(stderr, "PV::InitWeights::readWeights: problem reading arbor file %s, SHUTTING DOWN\n", arborfilename);
            exit(EXIT_FAILURE);
         }
         arbor += thisfilearbors;
      }  // while
   } // if useListOfArborFiles
   else if (combineWeightFiles){
      int rootproc = 0;
      int max_weight_files = 1;  // arbitrary limit...
      int num_weight_files = (int)conn->getParent()->parameters()->value(conn->getName(), "numWeightFiles", max_weight_files, true);
      int file_count=0;
      FILE * weightsfp = pvp_open_read_file(filename, icComm);
      if ((weightsfp == NULL) && (icComm->commRank() == rootproc) ){
         fprintf(stderr, ""
               "Cannot open file of weight files \"%s\".  Exiting.\n", filename);
         exit(EXIT_FAILURE);
      }

      char weightsfilename[PV_PATH_MAX];
      while( file_count < num_weight_files ) {
         if( icComm->commRank() == rootproc ) {
            char * fgetsstatus = fgets(weightsfilename, PV_PATH_MAX, weightsfp);
            if( fgetsstatus == NULL ) {
               bool endoffile = feof(weightsfp)!=0;
               if( endoffile ) {
                  fprintf(stderr, "File of weight files \"%s\" reached end of file before all %d weight files were read.  Exiting.\n", filename, num_weight_files);
                  exit(EXIT_FAILURE);
               }
               else {
                  int error = ferror(weightsfp);
                  assert(error);
                  fprintf(stderr, "File of weight files: error %d while reading.  Exiting.\n", error);
                  exit(error);
               }
            }
            else {
               // Remove linefeed from end of string
               weightsfilename[PV_PATH_MAX-1] = '\0';
               int len = strlen(weightsfilename);
               if (len > 1) {
                  if (weightsfilename[len-1] == '\n') {
                     weightsfilename[len-1] = '\0';
                  }
               }
            }
         } // commRank() == rootproc
         int filetype, datatype;
         int numParams = NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS;
         int params[NUM_BIN_PARAMS+NUM_WGT_EXTRA_PARAMS];
         pvp_read_header(weightsfilename, icComm, &timed, &filetype, &datatype, params, &numParams);
#ifndef USE_SHMGET
         int status = PV::readWeights(patches, dataStart, numArbors, numPatches, weightsfilename, icComm, &timed, preLoc);
#else
         int status = PV::readWeights(patches, dataStart, numArbors, numPatches, weightsfilename, icComm, &timed, preLoc, shmget_owner, shmget_flag);
#endif
         if (status != PV_SUCCESS) {
            fprintf(stderr, "PV::InitWeights::readWeights: problem reading arbor file %s, SHUTTING DOWN\n", weightsfilename);
            exit(EXIT_FAILURE);
         }
         file_count += 1;
      } // file_count < num_weight_files

   } // if combineWeightFiles
   else {
#ifndef USE_SHMGET
      int status = PV::readWeights(patches, dataStart, numArbors, numPatches, filename, icComm, &timed, preLoc);
#else
         int status = PV::readWeights(patches, dataStart, numArbors, numPatches, filename, icComm, &timed, preLoc, shmget_owner, shmget_flag);
#endif
      if (status != PV_SUCCESS) {
         fprintf(stderr, "PV::readWeights: problem reading weight file %s, SHUTTING DOWN\n", filename);
         exit(EXIT_FAILURE);
      }
      if( timef != NULL ) *timef = (float) timed;
   }
   return PV_SUCCESS;
}


/**
 * calculate gaussian weights between oriented line segments
 */
int InitWeights::gauss2DCalcWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitGauss2DWeightsParams * weightParamPtr) {



   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   float strength=weightParamPtr->getStrength();
   float aspect=weightParamPtr->getaspect();
   float shift=weightParamPtr->getshift();
   int numFlanks=weightParamPtr->getnumFlanks();
   float sigma=weightParamPtr->getsigma();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();
   double r2Max=weightParamPtr->getr2Max();
   double r2Min=weightParamPtr->getr2Min();
	
#ifndef USE_SHMGET	
   pvdata_t * w_tmp = dataStart;
#else
   volatile pvdata_t * w_tmp = dataStart;
#endif



   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);
      //TODO: add additional weight factor for difference between thPre and thPost
      if(weightParamPtr->checkTheta(thPost)) continue;
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = weightParamPtr->calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = weightParamPtr->calcXDelta(iPost);

            if(weightParamPtr->isSameLocOrSelf(xDelta, yDelta, fPost)) continue;

            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cos(thPost) + yDelta * sin(thPost);
            float yp = -xDelta * sin(thPost) + yDelta * cos(thPost);

            if(weightParamPtr->checkBowtieAngle(yp, xp)) continue;


            // include shift to flanks
            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = 0;
            if ((d2 <= r2Max) && (d2 >= r2Min)) {
               w_tmp[index] += strength*exp(-d2 / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if ((d2 <= r2Max) && (d2 >= r2Min)) {
                  w_tmp[index] += strength*exp(-d2 / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }

   return 0;
}



} /* namespace PV */
