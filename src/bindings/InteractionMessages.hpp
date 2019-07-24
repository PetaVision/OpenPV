#ifndef INTERACTIONMESSAGES_HPP_
#define INTERACTIONMESSAGES_HPP_

#include "observerpattern/BaseMessage.hpp"
#include "include/PVLayerLoc.h"
#include <vector>

namespace PV {

class InteractionMessage : public BaseMessage {
   public:
      InteractionMessage(std::string *errRef) {
         mError = errRef;
         *mError = "";
      }
      void error(std::string const err) const {
         if (*mError == "") {
            *mError = err;
         }
      }
   private:
      std::string *mError;
};

class LayerSetInternalStateMessage : public InteractionMessage {
  public:
   LayerSetInternalStateMessage(std::string *err, const char *name,
         std::vector<float> const *data):InteractionMessage(err) {
      setMessageType("LayerSetInternalState");
      mName = name;
      mData = data;
   }
   const char *mName;
   std::vector<float> const *mData;
};

class LayerGetInternalStateMessage : public InteractionMessage {
  public:
   LayerGetInternalStateMessage(std::string *err, const char *name,
         std::vector<float> *data):InteractionMessage(err) {
      setMessageType("LayerGetInternalState");
      mName = name;
      mData = data;
   }
   const char *mName;
   std::vector<float> *mData;
};

class LayerGetActivityMessage : public InteractionMessage {
  public:
   LayerGetActivityMessage(std::string *err, const char *name,
         std::vector<float> *data):InteractionMessage(err) {
      setMessageType("LayerGetActivity");
      mName = name;
      mData = data;
   }
   const char *mName;
   std::vector<float> *mData;
};

class LayerGetSparseActivityMessage : public InteractionMessage {
  public:
   LayerGetSparseActivityMessage(std::string *err, const char *name,
         std::vector<std::pair<float, int>> *data):InteractionMessage(err) {
      setMessageType("LayerGetSparseActivity");
      mName = name;
      mData = data;
   }
   const char *mName;
   std::vector<std::pair<float, int>> *mData;
};

class LayerGetShapeMessage : public InteractionMessage {
  public:
   LayerGetShapeMessage(std::string *err, const char *name,
         PVLayerLoc *loc):InteractionMessage(err) {
      setMessageType("LayerGetShape");
      mName = name;
      mLoc  = loc;
   }
   const char *mName;
   PVLayerLoc *mLoc;
};

class ProbeGetValuesMessage : public InteractionMessage {
  public:
   ProbeGetValuesMessage(std::string *err, const char *probeName,
         std::vector<double> *values):InteractionMessage(err) {
      setMessageType("ProbeGetValues");
      mName = probeName;
      mValues = values;
   }
   const char *mName;
   std::vector<double> *mValues;
};

class ConnectionGetPatchGeometryMessage : public InteractionMessage {
  public:
   ConnectionGetPatchGeometryMessage(std::string *err, const char *connName,
         int *nwp, int *nyp, int *nxp, int *nfp):InteractionMessage(err) {
      setMessageType("ConnectionGetPatchGeometry");
      mName = connName;
      mNwp  = nwp;
      mNyp  = nyp;
      mNxp  = nxp;
      mNfp  = nfp;
   }
   const char *mName;
   int *mNwp;
   int *mNyp;
   int *mNxp;
   int *mNfp;
};

class ConnectionSetWeightsMessage : public InteractionMessage {
  public:
   ConnectionSetWeightsMessage(std::string *err, const char *connName,
         std::vector<float> const *data):InteractionMessage(err) {
      setMessageType("ConnectionSetWeights");
      mName = connName;
      mData = data;
   }
   const char *mName;
   std::vector<float> const *mData;
};

class ConnectionGetWeightsMessage : public InteractionMessage {
  public:
   ConnectionGetWeightsMessage(std::string *err, const char *connName,
         std::vector<float> *data):InteractionMessage(err) {
      setMessageType("ConnectionGetWeights");
      mName = connName;
      mData = data;
   }
   const char *mName;
   std::vector<float> *mData;
};



} /* namespace PV */

#endif