/*
 * InitV.cpp
 *
 *  Created on: Dec 6, 2011
 *      Author: pschultz
 */

#include "InitV.hpp"

namespace PV {
InitV::InitV() {
   initialize_base();
   // Default constructor for derived classes.  A derived class should call
   // InitV::initialize from its own initialize routine, A derived class's
   // constructor should not call a non-default InitV constructor.
}

InitV::InitV(HyPerCol * hc, const char * groupName) {
   initialize_base();
   initialize(hc, groupName);
}

int InitV::initialize_base() {
   groupName = NULL;
   initVTypeCode = UndefinedInitV;
   filename = NULL;
   return PV_SUCCESS;
}

int InitV::initialize(HyPerCol * hc, const char * groupName) {
   int status = PV_SUCCESS;
   this->groupName = strdup(groupName);
   PVParams * params = hc->parameters();
   useStderr = hc->icCommunicator()->commRank()==0;
   const char * initVType = params->stringValue(groupName, "InitVType", true);
   if( initVType == NULL ) {
      initVTypeCode = ConstantV;
      constantValue = params->value(groupName, "valueV", V_REST);
      printerr("Using InitVType = \"ConstantV\" with valueV = %f\n", groupName, constantValue);
   }
   else if( !strcmp(initVType, "ConstantV") ) {
      initVTypeCode = ConstantV;
      constantValue = params->value(groupName, "valueV", V_REST);
   }
   else if( !strcmp(initVType, "ZeroV")) {
      initVTypeCode = ConstantV;
      constantValue = 0.0f;
   }
   else if( !strcmp(initVType, "UniformRandomV") ) {
      minV = params->value(groupName, "minV", 0.0f);
      maxV = params->value(groupName, "maxV", minV+1.0f);
      initVTypeCode = UniformRandomV;
   }
   else if( !strcmp(initVType, "GaussianRandomV") ) {
      meanV = params->value(groupName, "meanV", 0.0f);
      sigmaV = params->value(groupName, "sigmaV", 1.0f);
      initVTypeCode = GaussianRandomV;
   }
   else if( !strcmp(initVType, "InitVFromFile") ) {
      filename = params->stringValue(groupName, "Vfilename", true);
      if( filename == NULL ) {
         initVTypeCode = UndefinedInitV;
         printerr("InitV::initialize, group \"%s\": for InitVFromFile, string parameter \"Vfilename\" must be defined.  Exiting\n", groupName);
         abort();
      }
      initVTypeCode = InitVFromFile;
   }
   else {
      initVTypeCode = UndefinedInitV;
      printerr("InitV::initialize, group \"%s\": InitVType \"%s\" not recognized.\n", groupName, initVType);
      abort();
   }
   return status;
}

InitV::~InitV() {free(this->groupName);}

int InitV::calcV(HyPerLayer * layer) {
   int status = PV_SUCCESS;
   switch(initVTypeCode) {
   case UndefinedInitV:
      status = PV_FAILURE;
      printerr("InitV::calcV: InitVType was undefined.\n");
      break;
   case ConstantV:
      status = calcConstantV(layer->getV(), layer->getNumNeurons());
      break;
   case UniformRandomV:
      status = calcUniformRandomV(layer->getV(), layer->getNumNeurons());
      break;
   case GaussianRandomV:
      status = calcGaussianRandomV(layer->getV(), layer->getNumNeurons());
      break;
   case InitVFromFile:
      status = calcVFromFile(layer->getCLayer(), layer->getParent()->icCommunicator());
      break;
   default:
      status = PV_FAILURE;
      printerr("InitV::calcV: InitVType was an unrecognized type.\n");
      break;
   }
   return status;
}

int InitV::calcConstantV(pvdata_t * V, int numNeurons) {
   for( int k=0; k<numNeurons; k++ ) V[k] = constantValue;
   return PV_SUCCESS;
}

int InitV::calcGaussianRandomV(pvdata_t * V, int numNeurons) {
   for( int k=0; k<numNeurons; k++ ) V[k] = generateGaussianRand();
   return PV_SUCCESS;
}

#define GENERATEGAUSSIANRAND_TWOPI (6.283185307179586)
pvdata_t InitV::generateGaussianRand() {
   pvdata_t V;
   if( valueIsBeingHeld) {
      V = heldValue;
      valueIsBeingHeld = false;
   }
   else {
      double U1, U2;
      U1 = pv_random_prob();
      U2 = pv_random_prob();
      double t = GENERATEGAUSSIANRAND_TWOPI * U2;
      double r = sigmaV*sqrt(-2*log(U1)); // U1<1 so r is real
      V = r*cos(t) + meanV;
      heldValue = r*sin(t) + meanV;
      valueIsBeingHeld = true;
   }
   return V;

}

int InitV::calcUniformRandomV(pvdata_t * V, int numNeurons) {
   for( int k=0; k<numNeurons; k++ ) V[k] = generateUnifRand();
   return PV_SUCCESS;
}

pvdata_t InitV::generateUnifRand() {
   pvdata_t V = ( (pvdata_t) (pv_random()*uniformMultiplier) ) + minV;
   return V;
}

int InitV::calcVFromFile(PVLayer * clayer, InterColComm * icComm) {
   int status = PV_SUCCESS;
   const PVLayerLoc * loc = &(clayer->loc);
   pvdata_t * V = clayer->V;
   PVLayerLoc fileLoc;
   int filetype = getFileType(filename);
   if( filetype == PVP_FILE_TYPE) {
      int params[NUM_BIN_PARAMS];
      double timed;
      int pvpfiletype, datatype;
      int numParams = NUM_BIN_PARAMS;
      status = pvp_read_header(this->filename, icComm, &timed,
                          &pvpfiletype, &datatype, params, &numParams);
      assert(status == PV_SUCCESS);
      status = checkLoc(loc, params[INDEX_NX], params[INDEX_NY], params[INDEX_NF], params[INDEX_NX_GLOBAL], params[INDEX_NY_GLOBAL]);
      assert(status == PV_SUCCESS);
      fileLoc.nx = params[INDEX_NX];
      fileLoc.ny = params[INDEX_NY];
      fileLoc.nf = params[INDEX_NF];
      fileLoc.nb = params[INDEX_NB];
      fileLoc.nxGlobal = params[INDEX_NX_GLOBAL];
      fileLoc.nyGlobal = params[INDEX_NY_GLOBAL];
      switch(pvpfiletype) {
      case PVP_FILE_TYPE:
         status = read_pvdata(this->filename, icComm, &timed, V,
                              loc, PV_FLOAT_TYPE, false, false);
         break;
      case PVP_ACT_FILE_TYPE:
         printerr("calcVFromFile for file \"%s\": sparse activity files are not yet implemented.\n", this->filename);
         abort();
         break;
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         status = readNonspikingActFile(this->filename, icComm, &timed, V, params[INDEX_NBANDS]-1, &clayer->loc, datatype, false, false);
         break;
      default:
         printerr("calcVFromFile: file \"%s\" is not an activity pvp file.\n", this->filename);
         abort();
         break;
      }

   }
   else { // Treat as an image file
      status = getImageInfoGDAL(filename, icComm, &fileLoc, NULL);
      assert(status == PV_SUCCESS);
      if ( checkLoc(loc, fileLoc.nx, fileLoc.ny, fileLoc.nf, fileLoc.nxGlobal, fileLoc.nyGlobal)!=PV_SUCCESS ) {
         // error message produced by checkLoc
         abort();
      }
      int n=clayer->numNeurons;
      float * buf = new float[n];
      status = scatterImageFileGDAL(this->filename, 0, 0, icComm, &fileLoc, buf);
      // scatterImageFileGDAL handles the scaling by 1/255.0

      delete buf;
   }
   return status;
}

int InitV::checkLoc(const PVLayerLoc * loc, int nx, int ny, int nf, int nxGlobal, int nyGlobal) {
   int status = PV_SUCCESS;
   if( checkLocValue(loc->nxGlobal, nxGlobal, "nxGlobal") != PV_SUCCESS ) status = PV_FAILURE;
   if( checkLocValue(loc->nyGlobal, nyGlobal, "nyGlobal") != PV_SUCCESS ) status = PV_FAILURE;
   if( checkLocValue(loc->nf, nf, "nf") != PV_SUCCESS ) status = PV_FAILURE;
   return status;
}

int InitV::checkLocValue(int fromParams, int fromFile, const char * field) {
   int status = PV_SUCCESS;
   if( fromParams != fromFile ) {
      printerr("InitVFromFile: Incompatible %s: parameter group \"%s\" gives %d; filename \"%s\" gives %d\n",
               field, groupName, fromParams, filename, fromFile);
      status = PV_FAILURE;
   }
   return status;
}

int InitV::printerr(const char * fmtstring, ...) {
   int rtnval;
   va_list args;
   va_start(args, fmtstring);
   if( useStderr) {
      rtnval = vfprintf(stderr, fmtstring, args);
   }
   else {
      rtnval=0;
   }
   va_end(args);
   return rtnval;
}

}  // end namespace PV



