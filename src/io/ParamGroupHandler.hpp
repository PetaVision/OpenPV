/*
 * ParamGroupHandler.hpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

// Note: ParamGroupHandler and functions that depend on it were deprecated
// on March 24, 2016.  Instead, creating layers, connections, etc. should
// be handled using the PV_Init::registerKeyword, PV_Init::create, and
// PV_Init::build methods.

#ifndef PARAMGROUPHANDLER_HPP_
#define PARAMGROUPHANDLER_HPP_

#include <stddef.h>

namespace PV {

class HyPerCol;

typedef enum {
   UnrecognizedGroupType,
   HyPerColGroupType,
   LayerGroupType,
   ConnectionGroupType,
   ProbeGroupType,
   ColProbeGroupType, // TODO: make ColProbe a subclass of BaseProbe and eliminate this group type.
   WeightInitializerGroupType,
   WeightNormalizerGroupType
} ParamGroupType;

class HyPerCol;
class HyPerLayer;
class BaseConnection;
class ColProbe;
class BaseProbe;
class InitWeights;
class NormalizeBase;

class ParamGroupHandler {
public:
   ParamGroupHandler();
   virtual ~ParamGroupHandler() = 0;

   /**
    * Takes a keyword string as input, and returns whether the keyword is recognized as
    * a layer keyword, a connection keyword, etc.
    */
   virtual ParamGroupType getGroupType(char const * keyword) { return UnrecognizedGroupType; }

   /**
    * If the keyword is recognized as a HyPerCol keyword, return the HyPerCol object; otherwise return NULL.
    * Only the CoreParamGroupHandler should recognize a keyword returning HyPerColGroupType, and only the CoreParamGroupHandler
    * should return a non-NULL value.
    */
   virtual HyPerCol * createHyPerCol(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }

   /**
    * If the keyword is recognized as a layer keyword, create a layer of the appropriate type, with the given name and parent HyPerCol.
    * If the keyword is not recognized, or is a non-layer keyword, the function should return NULL.
    */
   virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }

   /**
    * If the keyword is recognized as a connection keyword, create a connection of the appropriate type,
    * with the given name, parent HyPerCol, weight initializer, and weight normalizer.
    * The weight initializer and weight normalizer arguments default to null.  Null values should be allowed in these two arguments
    * unless it is an error for the specified connection type to be instantiated with a null value.
    * If the keyword is not recognized, or is a non-layer keyword, the function should return NULL.
    */
   virtual BaseConnection * createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL) { return NULL; }

   /**
    * If the keyword is recognized as a ColProbe keyword, create a ColProbe of the appropriate type, with the given name and parent HyPerCol.
    * If the keyword is not recognized, or is a non-layer keyword, the function should return NULL.
    */
   virtual ColProbe * createColProbe(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }

   /**
    * If the keyword is recognized as a probe keyword, create a probe of the appropriate type, with the given name and parent HyPerCol.
    * This method should handle both layer probes and connection probes.
    * If the keyword is not recognized, or is a non-layer keyword, the function should return NULL.
    */
   virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }

   /**
    * If the keyword is recognized as a weight initialization keyword, create a weight initialization object
    * of the appropriate type, with the given name and parent HyPerCol.
    * If the keyword is not recognized, or is a non-layer keyword, the function should return NULL.
    */
   virtual InitWeights * createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }

   /**
    * If the keyword is recognized as a weight normalization keyword, create a weight normalization object
    * of the appropriate type, with the given name and parent HyPerCol.
    * If the keyword is not recognized, or is a non-layer keyword, the function should return NULL.
    */
   virtual NormalizeBase * createWeightNormalizer(char const * keyword, char const * name, HyPerCol * hc) { return NULL; }
};

} /* namespace PV */

#endif /* PARAMGROUPHANDLER_HPP_ */
