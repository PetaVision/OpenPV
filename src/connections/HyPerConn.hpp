/*
 * HyPerConnection.hpp
 *
 *  Created on: Oct 21, 2008
 *      Author: rasmussn
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "PVConnection.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

class HyPerCol;
class HyPerLayer;

extern PVConnParams defaultConnParams;

class HyPerConn {

   friend class HyPerCol;

public:
   HyPerConn();
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             int channel);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             const char * filename);
   HyPerConn(const char * name, int argc, char ** argv, HyPerCol * hc,
             HyPerLayer * pre, HyPerLayer * post);
   virtual ~HyPerConn();

   virtual int deliver(PVLayerCube * cube, int neighbor);

   virtual int updateWeights(PVLayerCube * preActivity, int neighbor);

   virtual int numberOfWeightPatches();
   virtual int writeWeights();

   virtual PVPatch * getWeights(int k, int bundle);
   virtual PVPatch * getPlasticityIncrement(int k, int bundle);

   inline PVPatch         ** weights()               {return wPatches;}
   inline PVSynapseBundle ** cliques()               {return bundles;}

   inline const char * getName()                     {return name;}
   inline int          getDelay()                    {return params->delay;}

   virtual PVSynapseBundle * tasks(int k, int neighbor)   {return bundles[k];}

   HyPerLayer * preSynapticLayer()     {return pre;}
   HyPerLayer * postSynapticLayer()    {return post;}

   // TODO - remove pvconn?
   // PVConnection * getCConnection()     {return pvconn;}

   int  getConnectionId()              {return connId;}
   void setConnectionId(int id)        {connId = id;}

   int setParams(PVParams * params, PVConnParams * p);

   int gauss2DCalcWeights(PVPatch * wp, int fPre, int xScale, int yScale,
                          int numFlanks, float flankShift, float rotate,
                          float aspect, float sigma, float r2Max, float strength);

protected:
   char * name;

   int connId;             // connection id
   int numBundles;         // number of synapse bundles
   int stdpFlag;           // presence of spike timing dependent plasticity
   float nxp, nyp, nfp;    // size of weight dimensions

   HyPerLayer    * pre;
   HyPerLayer    * post;
   HyPerCol      * parent;
   PVLayerCube   * pDecr;      // plasticity increment variable (Mi) for pre-synaptic layer
   PVPatch      ** pIncr;      // list of stdp patches Psij variable
   PVPatch      ** wPatches;   // list of weight patches
   PVSynapseBundle ** bundles; // list of tasks for each pre-synaptic neuron

   int numParams;
   PVConnParams * params;
   PVConnection * pvconn;

   int channel; // which channel of the post to update (e.g. inhibit)

protected:
   virtual int initialize(const char * filename, HyPerLayer * pre, HyPerLayer * post,
                          int channel);
   virtual int initializeWeights(const char * filename);
   virtual PVPatch ** createWeights();
   virtual int        deleteWeights();

   virtual int createSynapseBundles(int numTasks);
   virtual int adjustWeightBundles(int numTasks);
   int createNorthernSynapseBundles(int numTasks);

   // static member functions

public:
   static PVPatch ** createPatches(int numBundles, int nx, int ny, int nf)
   {
      PVPatch ** patches = (PVPatch**) malloc(numBundles*sizeof(PVPatch*));

      for (int i = 0; i < numBundles; i++) {
         patches[i] = pvpatch_new(nx, ny, nf);
      }

      return patches;
   }

   static int deletePatches(int numBundles, PVPatch ** patches)
   {
      for (int i = 0; i < numBundles; i++) {
         pvpatch_delete(patches[i]);
      }
      free(patches);

      return 0;
   }

};

}

#endif /* HYPERCONN_HPP_ */
