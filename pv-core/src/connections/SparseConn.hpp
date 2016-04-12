/*
 * SparseConn.hpp
 */

#ifndef SPARSECONN_HPP_
#define SPARSECONN_HPP_

#include "connections/HyPerConn.hpp"
#include <stdlib.h>
#include <vector>
#include <set>

namespace PV {

struct SparseWeightInfo {
   unsigned long size;
   pvwdata_t thresholdWeight;
   float percentile;
};

/**
 * A SparseConn identifies a connection between two layers
 */

class SparseConn : public HyPerConn {

public:

   SparseConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~SparseConn();

   virtual void deliverOnePreNeuronActivity(int patchIndex, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr);
   virtual void deliverOnePostNeuronActivity(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, taus_uint4 * rngPtr);
    
   int getNumWeightPatches() const {
      return numWeightPatches;
   }

   int getNumDataPatches() const {
      return numDataPatches;
   }


private:
   typedef pvwdata_t WeightType;
   typedef std::vector<WeightType> WeightListType;
   typedef std::vector<int> IndexListType;
   typedef HyPerConn super;
   typedef std::map<const WeightType * const, const WeightListType> WeightPtrMapType;
   typedef std::map<const WeightType * const, const IndexListType>  WeightIndexMapType;
   // Map nk -> weight ptr -> sparse weights
   typedef std::map<int, WeightPtrMapType> WeightMapType;
   // Map nk -> weight ptr -> output index
   typedef std::map<int, WeightIndexMapType> IndexMapType;

   float _sparsity;
   
#if 0
   WeightListType _sparseWeight;
   // The output offset into the post layer for a weight
   std::vector<int> _sparsePost;
   // Start of sparse weight data in the _sparseWeight array, indexed by data patch
   std::vector<int> _patchSparseWeightIndex;
   // Number of sparse weights for a patch, indexed by data patch
   std::vector<int> _patchSparseWeightCount;
#endif

   // Have sparse weights been allocated for each arbor?
   std::vector<bool> _sparseWeightsAllocated;

   WeightMapType _sparseWeightValues;
   IndexMapType _sparseWeightIndexes;
   SparseWeightInfo _sparseWeightInfo;

   std::set<int> _kPreExtWeightSparsified;

   unsigned long _numDeliverCalls; // Number of times deliver has been called
   unsigned long _allocateSparseWeightsFrequency; // Number of _numDeliverCalls that need to happen before the pre list needs to be rebuilt

   void allocateSparseWeightsPre(PVLayerCube const *activity, int arbor);
   void allocateSparseWeightsPost(PVLayerCube const *activity, int arbor);
   void calculateSparseWeightInfo();

   void ioParam_sparsity(enum ParamsIOFlag ioFlag);

protected:
   SparseConn();
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);
   int initialize_base();
   
   void allocateSparseWeights(const char *logPrefix);

   virtual int deliverPresynapticPerspective(PVLayerCube const * activity, int arborID);
   virtual int deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID);

protected:
};

} // namespace PV

#endif /* SPARSECONN_HPP_ */
