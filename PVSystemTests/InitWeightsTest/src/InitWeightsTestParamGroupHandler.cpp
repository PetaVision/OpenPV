/*
 * InitWeightsTestParamGroupHandler.cpp
 *
 *  Created on: Feb 13, 2015
 *      Author: pschultz
 */

#include "InitWeightsTestParamGroupHandler.hpp"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "HyPerConnDebugInitWeights.hpp"
#include "KernelConnDebugInitWeights.hpp"
#include "InitWeightTestProbe.hpp"
#include "InitGaborWeights.hpp"

namespace PV {

InitWeightsTestParamGroupHandler::InitWeightsTestParamGroupHandler() {
}

ParamGroupType InitWeightsTestParamGroupHandler::getGroupType(char const * keyword) {
   if (keyword==NULL) { return UnrecognizedGroupType; }
   else if (!strcmp(keyword, "HyPerConnDebugInitWeights")) { return ConnectionGroupType; }
   else if (!strcmp(keyword, "KernelConnDebugInitWeights")) { return ConnectionGroupType; }
   else if (!strcmp(keyword, "InitWeightTestProbe")) { return ProbeGroupType; }
   else if (!strcmp(keyword, "GaborWeight")) { return WeightInitializerGroupType; }
   else { return UnrecognizedGroupType; }
}

BaseConnection * InitWeightsTestParamGroupHandler::createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   BaseConnection * addedConnection = NULL;
   bool recognized = false;
   if (keyword==NULL) {
      addedConnection = NULL;
   }
   else if( !strcmp(keyword, "HyPerConnDebugInitWeights") ) {
      recognized = true;
      addedConnection = new HyPerConnDebugInitWeights(name, hc, weightInitializer, weightNormalizer);
   }
   else if( !strcmp(keyword, "KernelConnDebugInitWeights") ) {
      recognized = true;
      addedConnection = new KernelConnDebugInitWeights(name, hc, weightInitializer, weightNormalizer);
   }
   else {
      assert(!recognized);
   }

   if (recognized && addedConnection==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "createConnection error: unable to add %s\n", keyword);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return addedConnection;
}

BaseProbe * InitWeightsTestParamGroupHandler::createProbe(char const * keyword, char const * name, HyPerCol * hc) {
   BaseProbe * addedProbe = NULL;
   bool recognized = false;
   if (keyword==NULL) {
      addedProbe = NULL;
   }
   else if (!strcmp(keyword, "InitWeightTestProbe")) {
      recognized = true;
      addedProbe = new InitWeightTestProbe(name, hc);
   }
   else {
      assert(!recognized);
   }

   if (recognized && addedProbe==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "createProbe error: unable to add %s\n", keyword);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return addedProbe;
}

InitWeights * InitWeightsTestParamGroupHandler::createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc) {
   InitWeights * weightInitializer = NULL;
   bool recognized = false;
   if (keyword==NULL) {
      weightInitializer = NULL;
   }
   else if (!strcmp(keyword, "GaborWeight")) {
      recognized = true;
      weightInitializer = new InitGaborWeights(name, hc);
   }

   return weightInitializer;

}

InitWeightsTestParamGroupHandler::~InitWeightsTestParamGroupHandler() {
}

} /* namespace PV */
