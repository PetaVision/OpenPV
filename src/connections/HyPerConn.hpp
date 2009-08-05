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
class ConnectionProbe;

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

   virtual int insertProbe(ConnectionProbe * p);
   virtual int outputState(float time);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(PVLayerCube * preActivity, int neighbor);

   inline  int numberOfBundles()                     {return numBundles;}
   virtual int numberOfWeightPatches();
   virtual int writeWeights();
   virtual int writeWeights(int k);
   virtual int writeWeights(const char * filename, int k);

   virtual PVPatch * getWeights(int k, int bundle);
   virtual PVPatch * getPlasticityIncrement(int k, int bundle);

   inline PVLayerCube * getPlasticityDecrement()     {return pDecr;}

   inline PVPatch         ** weights()               {return wPatches;}
   inline PVSynapseBundle ** cliques()               {return bundles;}

   inline const char * getName()                     {return name;}
   inline int          getDelay()                    {return params->delay;}

   inline float minWeight()                          {return 0.0;}
   inline float maxWeight()                          {return wMax;}

   virtual PVSynapseBundle * tasks(int k, int neighbor)   {return bundles[k];}

   HyPerLayer * preSynapticLayer()     {return pre;}
   HyPerLayer * postSynapticLayer()    {return post;}

   // TODO - remove pvconn?
   // PVConnection * getCConnection()     {return pvconn;}

   int  getConnectionId()              {return connId;}
   void setConnectionId(int id)        {connId = id;}

   int setParams(PVParams * params, PVConnParams * p);

   PVPatch ** convertPreSynapticWeights(float time);

   int randomWeights(PVPatch * wp, float wMin, float wMax, int seed);

   int gauss2DCalcWeights(PVPatch * wp, int fPre, int no, int xScale, int yScale,
                          int numFlanks, float flankShift, float rotate,
                          float aspect, float sigma, float r2Max, float strength);

protected:
   char * name;

   int connId;             // connection id
   int numBundles;         // number of synapse bundles
   int stdpFlag;           // presence of spike timing dependent plasticity
   float nxp, nyp, nfp;    // size of weight dimensions

   // STDP parameters for modifying weights
   float ampLTP;  // long term potentiation amplitude
   float ampLTD;  // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float dWMax;
   float wMax;

   HyPerLayer    * pre;
   HyPerLayer    * post;
   HyPerCol      * parent;
   PVLayerCube   * pDecr;      // plasticity decrement variable (Mi) for pre-synaptic layer
   PVPatch      ** pIncr;      // list of stdp patches Psij variable
   PVPatch      ** wPatches;   // list of weight patches
   PVPatch      ** wPostPatches;  // post-synaptic linkage of weights
   PVSynapseBundle ** bundles; // list of tasks for each pre-synaptic neuron

   int numParams;
   PVConnParams * params;
   PVConnection * pvconn;

   int numProbes;
   ConnectionProbe ** probes;   // probes used to output data

   int channel; // which channel of the post to update (e.g. inhibit)
   int ioAppend;                // controls opening of binary files
   float wPostTime;             // time of last conversion to wPostPatches

protected:
   virtual int initialize(const char * filename, HyPerLayer * pre, HyPerLayer * post,
                          int channel);
   virtual int initializeWeights(const char * filename);
   virtual int initializeRandomWeights(int seed);
   virtual PVPatch ** createWeights(int nxPatch, int nyPatch, int nfPatch, int numPatches);
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
         patches[i] = pvpatch_inplace_new(nx, ny, nf);
      }

      return patches;
   }

   static int deletePatches(int numBundles, PVPatch ** patches)
   {
      for (int i = 0; i < numBundles; i++) {
         pvpatch_inplace_delete(patches[i]);
      }
      free(patches);

      return 0;
   }

};

}

#endif /* HYPERCONN_HPP_ */
