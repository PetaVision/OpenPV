/**
 * MLearningGroupHandler.cpp
 *
 * Created on: Mar 5, 2015
 *     Author: pschultz
 */

#include "MLearningGroupHandler.hpp"
#include "GradientCheckConn.hpp"
#include "MLPErrorLayer.hpp"
#include "MLPForwardLayer.hpp"
#include "MLPSigmoidLayer.hpp"
#include "MLPOutputLayer.hpp"

namespace PVMLearning{

PV::ParamGroupType MLearningGroupHandler::getGroupType(char const * keyword) {
   PV::ParamGroupType result = PV::UnrecognizedGroupType;
   if (keyword == NULL) { return result; }
   else if (!strcmp(keyword, "MLPErrorLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "MLPForwardLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "MLPSigmoidLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "MLPOutputLayer")) { result = PV::LayerGroupType; }
   else if (!strcmp(keyword, "GradientCheckConn")) { result = PV::ConnectionGroupType; }

   return result;
}

PV::HyPerLayer * MLearningGroupHandler::createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
   PV::HyPerLayer * layer = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::LayerGroupType) { return layer; }
   else if (!strcmp(keyword, "MLPErrorLayer")) { layer = new MLPErrorLayer(name, hc); }
   else if (!strcmp(keyword, "MLPForwardLayer")) { layer = new MLPForwardLayer(name, hc); }
   else if (!strcmp(keyword, "MLPSigmoidLayer")) { layer = new MLPSigmoidLayer(name, hc); }
   else if (!strcmp(keyword, "MLPOutputLayer")) { layer = new MLPOutputLayer(name, hc); }
   
   if (layer==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return layer;
}

PV::BaseConnection * MLearningGroupHandler::createConnection(char const * keyword, char const * name, PV::HyPerCol * hc, PV::InitWeights * weightInitializer, PV::NormalizeBase * weightNormalizer){
   PV::BaseConnection * connection = NULL;
   if (keyword == NULL || getGroupType(keyword) != PV::ConnectionGroupType) { return connection; }

   else if (!strcmp(keyword, "GradientCheckConn")) { connection = new GradientCheckConn(name, hc); }
   if (connection ==NULL) {
      fprintf(stderr, "Rank %d process unable to create %s \"%s\"\n", hc->columnId(), keyword, name);
      exit(EXIT_FAILURE);
   }
   return connection;
}

}  // namespace PVMatchingPursuit
