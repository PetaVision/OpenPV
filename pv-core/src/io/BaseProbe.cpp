/*
 * BaseProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "BaseProbe.hpp"
#include "ColumnEnergyProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

BaseProbe::BaseProbe()
{
   initialize_base();
   // Derived classes of BaseProbe should call BaseProbe::initialize themselves.
}

BaseProbe::~BaseProbe()
{
   if (outputstream != NULL) {
      PV_fclose(outputstream); outputstream = NULL;
   }
   free(targetName); targetName = NULL;
   free(msgparams); msgparams = NULL;
   free(msgstring); msgstring = NULL;
   free(probeOutputFilename); probeOutputFilename = NULL;
   if(triggerLayerName){
      free(triggerLayerName);
      triggerLayerName = NULL;
   }
   free(name);
   free(energyProbe);
}

int BaseProbe::initialize_base() {
   name = NULL;
   parent = NULL;
   owner = NULL;
   outputstream = NULL;
   targetName = NULL;
   msgparams = NULL;
   msgstring = NULL;
   probeOutputFilename = NULL;
   triggerFlag = false;
   triggerLayerName = NULL;
   triggerLayer = NULL;
   triggerOffset = 0;
   energyProbe = NULL;
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
int BaseProbe::initialize(const char * probeName, HyPerCol * hc)
{
   setParentCol(hc);
   setProbeName(probeName);
   ioParams(PARAMS_IO_READ);
   //Add probe to list of probes
   parent->addBaseProbe(this); // Adds probe to HyPerCol.  If needed, probe will be attached to layer or connection during communicateInitInfo
   owner = (void *) parent;
   return PV_SUCCESS;
}

int BaseProbe::setProbeName(const char * probeName) {
   assert(this->name == NULL);
   this->name = strdup(probeName);
   if (this->name == NULL) {
      assert(parent!=NULL);
      fprintf(stderr,"BaseProbe \"%s\" unable to set probeName on rank %d: %s\n",
            probeName, parent->columnId(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

char const * BaseProbe::getKeyword() {
   return this->getParent()->parameters()->groupKeywordFromName(this->getName());
}

int BaseProbe::ioParams(enum ParamsIOFlag ioFlag) {
   parent->ioParamsStartGroup(ioFlag, name);
   ioParamsFillGroup(ioFlag);
   parent->ioParamsFinishGroup(ioFlag);
   return PV_SUCCESS;
}

int BaseProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_targetName(ioFlag);
   ioParam_message(ioFlag);
   ioParam_probeOutputFile(ioFlag);
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_energyProbe(ioFlag);
   ioParam_coefficient(ioFlag);
   return PV_SUCCESS;
}

void BaseProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "targetName", &targetName);
}

void BaseProbe::ioParam_message(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "message", &msgparams, NULL, false/*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      initMessage(msgparams);
   }
}

void BaseProbe::ioParam_energyProbe(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "energyProbe", &energyProbe, NULL, false/*warnIfAbsent*/);
}

void BaseProbe::ioParam_coefficient(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "energyProbe"));
   if (energyProbe && energyProbe[0]) {
      parent->ioParamValue(ioFlag, name, "coefficient", &coefficient, 1.0/*default*/, true/*warnIfAbsent*/);
   }
}

void BaseProbe::ioParam_probeOutputFile(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "probeOutputFile", &probeOutputFilename, NULL, false/*warnIfAbsent*/);
}

void BaseProbe::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "triggerFlag", &triggerFlag, triggerFlag);
}

void BaseProbe::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerFlag"));
   if (triggerFlag) {
      parent->ioParamStringRequired(ioFlag, name, "triggerLayerName", &triggerLayerName);
   }
}

void BaseProbe::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerFlag"));
   if (triggerFlag) {
      parent->ioParamValue(ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
      if(triggerOffset < 0){
         fprintf(stderr, "%s \"%s\" error in rank %d process: TriggerOffset (%f) must be positive\n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId(), triggerOffset);
         exit(EXIT_FAILURE);
      }
   }
}

int BaseProbe::initOutputStream(const char * filename) {
   if( parent->columnId()==0 ) {
      if( filename != NULL ) {
         char * outputdir = parent->getOutputPath();
         char * path = (char *) malloc(strlen(outputdir)+1+strlen(filename)+1);
         sprintf(path, "%s/%s", outputdir, filename);
         bool append = parent->getCheckpointReadFlag();
         const char * fopenstring = append ? "a" : "w";
         outputstream = PV_fopen(path, fopenstring, parent->getVerifyWrites());
         if( !outputstream ) {
            fprintf(stderr, "BaseProbe error opening \"%s\" for writing: %s\n", path, strerror(errno));
            exit(EXIT_FAILURE);
         }
         free(path);
      }
      else {
         outputstream = PV_stdout();
      }
   }
   else {
      outputstream = NULL; // Only root process writes; if other processes need something written it should be sent to root.
                           // Derived classes for which it makes sense for a different process to do the file i/o should override initOutputStream
   }
   return PV_SUCCESS;
}

int BaseProbe::communicateInitInfo() {
   //Set up output stream
   int status = initOutputStream(probeOutputFilename);
   //communicate here only sets up triggering
   if(triggerFlag){
      triggerLayer = parent->getLayerFromName(triggerLayerName);
      if (triggerLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
                  parent->parameters()->groupKeywordFromName(name), name, triggerLayerName);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
   }
   if (energyProbe && energyProbe[0]) {
      ColProbe * colprobe = getParent()->getColProbeFromName(energyProbe);
      ColumnEnergyProbe * probe = dynamic_cast<ColumnEnergyProbe *>(colprobe);
      if (probe==NULL) {
         if (getParent()->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: energyProbe \"%s\" is not a ColumnEnergyProbe in the column.\n",
                  getParent()->parameters()->groupKeywordFromName(getName()), getName(), energyProbe);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(getParent()->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
      status = probe->addTerm(this, coefficient, getParent()->getNBatch());
   }
   return status;
}

int BaseProbe::allocateDataStructures(){
   return PV_SUCCESS;
}

int BaseProbe::initMessage(const char * msg) {
   assert(msgstring==NULL);
   int status = PV_SUCCESS;
   if( msg != NULL && msg[0] != '\0' ) {
      size_t msglen = strlen(msg);
      this->msgstring = (char *) calloc(msglen+2, sizeof(char)); // Allocate room for colon plus null terminator
      if(this->msgstring) {
         memcpy(this->msgstring, msg, msglen);
         this->msgstring[msglen] = ':';
         this->msgstring[msglen+1] = '\0';
      }
   }
   else {
      this->msgstring = (char *) calloc(1, sizeof(char));
      if(this->msgstring) {
         this->msgstring[0] = '\0';
      }
   }
   if( !this->msgstring ) {
      fprintf(stderr, "%s \"%s\": Unable to allocate memory for probe's message.\n",
            parent->parameters()->groupKeywordFromName(name), name);
      status = PV_FAILURE;
   }
   assert(status == PV_SUCCESS);
   return status;
}

bool BaseProbe::needUpdate(double time, double dt){
   if(triggerFlag){
      assert(triggerLayer);
      double updateTime;
      //Update if trigger layer updated on this timestep
      if(fabs(time - triggerLayer->getLastUpdateTime()) <= (dt/2)){
         updateTime = triggerLayer->getLastUpdateTime();
      }
      else{
         updateTime = triggerLayer->getNextUpdateTime();
      }
      //never update flag
      if(updateTime == -1){
         return false;
      }
      //Check for equality
      if(fabs(time - (updateTime - triggerOffset)) < (dt/2)){
         return true;
      }
      //If it gets to this point, don't update
      return false;
   }
   //If no trigger, update every timestep
   else{
      return true;
   }
}

int BaseProbe::outputStateWrapper(double timef, double dt){
   int status = PV_SUCCESS;
   if(needUpdate(timef, dt)){
      status = outputState(timef);
   }
   return status;
}

/**
 * @time
 */
//int BaseProbe::outputState(double timef)
//{
//   return 0;
//}

} // namespace PV
