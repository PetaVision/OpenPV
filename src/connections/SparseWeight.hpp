#ifndef SPARSE_WEIGHT_H_
#define SPARSE_WEIGHT_H_

#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <sstream>
#include <ctime>

#include "connections/HyPerConn.hpp"
#include "include/pv_types.h"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

// Everything is included in the .hpp file because this code meant to become heavily templated in
// the future, and C++ makes placing templated functions in the the .cpp file cumbersome.

namespace PV {

/**
 * Base class for weight data
 */
class WeightData : public BaseObject {
public:
   typedef float Type; //< This will eventually be a template parameter

   /**
    * Constructor
    *
    * @param conn  The HyPerConn for these sparse weights
    */
   WeightData(HyPerConn * conn)
   : BaseObject()
   , mConn(conn)
   {
   }

   /**
    * Destructor
    */
   virtual ~WeightData() {}

   //------------------------------------------------------------------------------------
   // HyPerConn dependencies go here. As refactoring occurs, maybe some of these
   // dependencies will disappear. Regardless, the idea is to be *very* clear about
   // the ways in which the weight data structure depends on HyPerConn
   //------------------------------------------------------------------------------------

   /**
    * \defgroup HyPerConn dependencies
    * @{
    */

   /**
    * @return the number of extended presynaptic neurons in the connection
    */
   int numPreSynapticLayerExtendedNeurons() const {
      return mConn->preSynapticLayer()->getNumExtended();
   }

   /**
    * @return the connection's number of axonal arbor lists
    */
   int numAxonalArborLists() const {
      return mConn->numberOfAxonalArborLists();
   }

   /**
    * @return number of data patches in the connection
    */
   int numDataPatches() const {
      return mConn->getNumDataPatches();
   }

   /**
    * @return the connection's F patch size
    */
   int fPatchSize() const {
      return mConn->fPatchSize();
   }

   /**
    * @return the connection's X patch size
    */
   int xPatchSize() const {
      return mConn->xPatchSize();
   }

   /**
    * @return the connection's Y patch size
    */
   int yPatchSize() const {
      return mConn->yPatchSize();
   }

   /**
    * @return the connection's weight data
    */
   Type * weightData(int arborId, int patchIndex) {
      return mConn->get_wData(arborId, patchIndex);
   }

   /**
    * @return the connection's Y patch stride
    */
   int yPatchStride() const {
      return mConn->yPatchStride();
   }

   /**
    * @return the connection's weight patch descriptors
    */
   PVPatch * getWeights(int k, int arbor) {
      return mConn->getWeights(k, arbor);
   }

   /**
    * @return the connection's target location descriptor
    */
   const PVLayerLoc * targetLoc() {
      return mConn->post->getLayerLoc();
   }

   /**
    * @return the connection's source location descriptor
    */
   const PVLayerLoc * sourceLoc() {
      return mConn->preSynapticLayer()->getLayerLoc();
   }

   /**
    * @return the connection's weight data
    */
   Type ** weightData() const {
      return mConn->get_wDataStart();
   }

   /**
    * @return the connection's num location descriptor
    */
   int getNumPostNeurons() const {
      return mConn->post->getNumNeurons();
   }

   /**
    * @return the post connection's Y patch stride
    */
   int postConnYPatchStride() const {
      return mConn->postConn->yPatchStride();
   }

   /**
    * @return post connection's Y patch size
    */
   int postConnYPatchSize() const {
      return mConn->postConn->yPatchSize();
   }

   /**
    * @return the post connection's X patch stride
    */
   int postConnXPatchStride() const {
      return mConn->postConn->xPatchStride();
   }

   /**
    * @return post connection's X patch size
    */
   int postConnXPatchSize() const {
      return mConn->postConn->xPatchSize();
   }

   /**
    * @return the post connection's F patch stride
    */
   int postConnFPatchStride() const {
      return mConn->postConn->fPatchStride();
   }

   /**
    * @return post connection's F patch size
    */
   int postConnFPatchSize() const {
      return mConn->postConn->fPatchSize();
   }

   /**
    * @return the post connnection's patch to data lookup
    */
   int postConnPatchToDataLUT(int kTargetExt) const {
      return mConn->postConn->patchToDataLUT(kTargetExt);
   }

   /**
    * @return the post connection's weight data head
    */
   const Type * const postConnWDataHead(int arbor, int kernel) const {
      return mConn->postConn->get_wDataHead(arbor, kernel);
   }

protected:
   HyPerConn * mConn; //< The HyPerConn for which these weights are used

   // End of HyPerConn dependencies
   /**@}*/
};

/**
 * Type traits for the SparseWeight classes
 *
 * These traits can be easily imported into a class using
 * the SPARSE_WEIGHT_TYPEDEFS(T) macro
 *
 * Replace T with a built-in type like float or int
 */
template<typename T>
struct SparseWeightTraits {
   typedef T WeightType;
   typedef std::vector<WeightType> WeightListType;
   typedef std::vector<int> IndexListType;

   typedef std::map<const WeightType * const, const WeightListType> WeightPtrMapType;
   typedef std::map<const WeightType * const, const IndexListType>  WeightIndexMapType;

   // Map nk -> weight ptr -> sparse weights
   typedef std::map<int, WeightPtrMapType> WeightMapType;
   // Map nk -> weight ptr -> output index
   typedef std::map<int, WeightIndexMapType> IndexMapType;
};

// Helper macro for bringing in sparse weight types into a class
// T is expected to be a built-in type, like float or int
#define SPARSE_WEIGHT_TYPEDEFS(T) \
   typedef typename ::PV::SparseWeightTraits<T>::WeightType WeightType; \
   typedef typename ::PV::SparseWeightTraits<T>::WeightListType WeightListType; \
   typedef typename ::PV::SparseWeightTraits<T>::IndexListType IndexListType; \
   typedef typename ::PV::SparseWeightTraits<T>::WeightPtrMapType WeightPtrMapType; \
   typedef typename ::PV::SparseWeightTraits<T>::WeightMapType WeightMapType; \
   typedef typename ::PV::SparseWeightTraits<T>::WeightIndexMapType WeightIndexMapType; \
   typedef typename ::PV::SparseWeightTraits<T>::IndexMapType IndexMapType; \

/**
 *
 * Sparse weight data structure
 *
 * This is a proof-of-concept implementation of sparse weights, where only the top N percent
 * of weights are stored and used in the convolution.
 *
 * This data strcuture maps a pointer that is used in deliverOne(Pre|Post)NeuronActivity, and the
 * number of elements to a list of sparse weights. This can be a bit redundant, but the goal is 
 * to show that sparse weights are useful before implementing an optimized solution.
 */
class SparseWeight : public WeightData {
public:
   SPARSE_WEIGHT_TYPEDEFS(float)

   /**
    * Constructor for the sparse weight data structure. The sparse weights are not allocated
    * when the constructor is called. You will need to call refresh()
    *
    * @param conn                The HyPerConn that is using these sparse weights. 
    * @param percentile          Weight sparsity percentile. Only weights whose magnitude are at this
    *                            percentile or greater are included in the list of sparse weights
    */
   SparseWeight(HyPerConn * conn, float percentile)
   : WeightData(conn)
   , mPercentile(percentile)
   {
      pvAssert(mPercentile >= 0.0);
      pvAssert(mPercentile <= 1.0);
   }

   // Copies not allowed
   SparseWeight(const SparseWeight&) = delete;
   SparseWeight& operator=(const SparseWeight&) = delete;

   /**
    * Refresh the set of sparse weights from the weight data.
    */
   virtual void refresh() {
      clock_t begin = std::clock();
      mValues.clear();
      mIndexes.clear();
      allocate(numPreSynapticLayerExtendedNeurons(), mPercentile, weightData(), numAxonalArborLists(), numDataPatches(), fPatchSize(), xPatchSize(), yPatchSize());
      clock_t end = std::clock();
      pvInfo() << getName() << ": allocated in " << double(end - begin) / CLOCKS_PER_SEC << " seconds " << std::endl;
   }

   /**
    * @param nk  Size of the convolution's inner loop using the full set of weights
    * @param ptr The pointer to the full set of weights
    *
    * @return The list of sparse weights
    */
   const WeightListType& values(int nk, const WeightType * const ptr) const {
      pvAssert(mValues.find(nk) != mValues.end());
      pvAssert(mValues.find(nk)->second.find(ptr) != mValues.find(nk)->second.end());
      return mValues.find(nk)->second.find(ptr)->second;
   }

   /**
    * @param nk  Size of the convolution's inner loop using the full set of weights
    * @param ptr The pointer to the full set of weights
    *
    * @return The output indexes into the post-synaptic layer
    */
   const IndexListType& indexes(int nk, const WeightType * const ptr) const {
      pvAssert(mIndexes.find(nk) != mIndexes.end());
      pvAssert(mIndexes.find(nk)->second.find(ptr) != mIndexes.find(nk)->second.end());
      return mIndexes.find(nk)->second.find(ptr)->second;
   }

protected:

   /**
    * Find the weight that matches the percentile
    *
    * @param percentile          Weight sparsity percentile. Only weights whose magnitude are at this
    *                            percentile or greater are included in the list of sparse weights
    * @param wData               Non-sparse weight data, seperate list for each axonal arbor list
    * @param numAxonalArborLists Number of axonal arbor lists
    * @param numPatches          Number of weight patches
    * @param nfp                 Number of features in a weight patch
    * @param nxp                 X dimension size of a weight patch
    * @param nyp                 Y dimension size of a weight patch
    *
    * @return std::pair: (smallest weight, number of sparse weights)
    */
   std::pair<WeightType, size_t>
   findWeightThreshold(float percentile, WeightType ** wData, size_t numAxonalArborLists, size_t numPatches, int nfp, int nxp, int nyp) const {

      pvAssert(percentile >= 0.0f);
      pvAssert(percentile <= 1.0f);

      size_t patchSize = nfp * nxp * nyp;
      size_t fullWeightSize = numAxonalArborLists * numPatches * patchSize;

      if (percentile >= 1.0) {
         return std::make_pair(1.0, fullWeightSize);
      }

      std::vector<WeightType> weights;
      weights.reserve(fullWeightSize);

      for (int ar = 0; ar < numAxonalArborLists; ar++) {
         for (int pt = 0; pt < numPatches; pt++) {
            // Why this would ever happen is inexplicable.
            if (wData[ar] == nullptr) {
               pvError() << getName() << ": wData[" << ar << "] is null" << std::endl;
            }
            pvAssert(wData[ar] != nullptr);
            WeightType *weight = &wData[ar][pt * patchSize];
            for (int k = 0; k < patchSize; k++) {
               weights.push_back(fabs(weight[k]));
            }
         }
      }

      std::sort(weights.begin(), weights.end());
      int index = weights.size() * percentile;
      return std::make_pair(weights[index], weights.size() - index);
   }

   virtual void allocate(const int numPreExt, float percentile, WeightType ** wData, size_t numAxonalArborLists, size_t numPatches, int nfp, int nxp, int nyp ) = 0;

protected:
   float mPercentile;         //< Weight sparsity percentile. Only weights whose magnitude
                              //<   are at this percentile or greater are included in the
                              //<   list of sparse weights
   WeightMapType mValues;     //< Map nk -> weight ptr -> sparse weights
   IndexMapType mIndexes;     //< Map nk -> weight ptr -> output index
};

/**
 * Sparse weights used for delivery from the pre-synaptic perspective
 */
class SparseWeightPre : public SparseWeight {
public:
   /**
    * Constructor for the sparse weight data structure. The sparse weights are not allocated
    * when the constructor is called. You will need to call refresh()
    *
    * @param conn          The HyPerConn that is using these sparse weights.
    * @param percentile    Weight sparsity percentile. Only weights whose magnitude are at this
    *                      percentile or greater are included in the list of sparse weights
    */
   SparseWeightPre(HyPerConn * conn, float percentile)
   : SparseWeight(conn, percentile)
   {
      std::stringstream name;
      name << conn->getName() << ".SparseWeightPre";
      setName(name.str().c_str());
   }

protected:
   /**
    * Allocate sparse weights for the pre-synaptic perspective
    *
    * @param numPreExt Number of pre-extended neurons
    * @param percentile The weight percentile, between 0.0 and 1.0
    * @param wDataStart The data patches
    * @param numAxonalArborLists Number of axonal arbor lists
    * @param numPatches Number of data patches
    * @param nfp Number of features in a patch
    * @param nxp X size dimension of the data patches
    * @param nyp Y size dimension of the data patches
    */
   virtual void allocate(const int numPreExt, float percentile, WeightType ** wData, size_t numAxonalArborLists, size_t numPatches, int nfp, int nxp, int nyp) {

      WeightType threshold;
      size_t size;
      std::tie(threshold, size) = findWeightThreshold(percentile, wData, numAxonalArborLists, numPatches, nfp, nxp, nyp);

      for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
         for (int kPreExt = 0; kPreExt < numPreExt; kPreExt++) {
            PVPatch *patch = getWeights(kPreExt, arbor);
            const int nk = patch->nx * nfp;
            const int nyp = patch->ny;
            const WeightType * const weightDataStart = weightData(arbor, kPreExt);

            for (int y = 0; y < nyp; y++) {
               const WeightType * const weightPtr = weightDataStart + y * yPatchStride();

               // Don't re-sparsify something that's already been put thru the sparsfication grinder
               bool shouldSparsify = false;

               // Find the weight pointers for this nk sized patch
               typename WeightMapType::iterator sparseWeightValuesNk = mValues.find(nk);
               typename IndexMapType::iterator sparseWeightIndexesNk = mIndexes.find(nk);

               if (sparseWeightValuesNk == mValues.end()) {
                  // Weight pointers don't exist for this sized nk. Allocate a map for this nk
                  mValues.insert(make_pair(nk, WeightPtrMapType()));
                  mIndexes.insert(make_pair(nk, WeightIndexMapType()));
                  // Get references
                  sparseWeightValuesNk = mValues.find(nk);
                  sparseWeightIndexesNk = mIndexes.find(nk);
                  shouldSparsify = true;
               } else if (sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
                  // This nk group exists, but no weight pointer.
                  shouldSparsify = true;
               }

               if (shouldSparsify) {
                  WeightListType sparseWeight;
                  IndexListType idx;

                  // Equivalent to inner loop accumulate
                  for (int k = 0; k < nk; k++) {
                     WeightType weight = weightPtr[k];
                     if (std::abs(weight) >= threshold) {
                        sparseWeight.push_back(weight);
                        idx.push_back(k);
                     }
                  }

                  sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
                  sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
               }
            }
         }
      }
   }
};

/**
 * Sparse weights used for delivery from the post-synaptic perspective
 */
class SparseWeightPost : public SparseWeight {
public:
   /**
    * Constructor for the sparse weight data structure. The sparse weights are not allocated
    * when the constructor is called. You will need to call refresh()
    *
    * @param conn                The HyPerConn that is using these sparse weights.
    * @param percentile          Weight sparsity percentile. Only weights whose magnitude are at this
    *                            percentile or greater are included in the list of sparse weights
    */
   SparseWeightPost(HyPerConn * conn, float percentile)
   : SparseWeight(conn, percentile)
   {
      std::stringstream sparseName;
      sparseName << conn->getName() << ".SparseWeightPost";
      setName(sparseName.str().c_str());
   }

protected:
   /**
    * Allocate sparse weights for the post-synaptic perspective
    *
    * @param numPreExt Number of pre-extended neurons
    * @param percentile The weight percentile, between 0.0 and 1.0
    * @param wDataStart The data patches
    * @param numAxonalArborLists Number of axonal arbor lists
    * @param numPatches Number of data patches
    * @param nfp Number of features in a patch
    * @param nxp X size dimension of the data patches
    * @param nyp Y size dimension of the data patches
    */
   void allocate(const int numPreExt, float percentile, WeightType ** wData, size_t numAxonalArborLists, size_t numPatches, int nfp, int nxp, int nyp) {
      WeightType threshold;
      size_t size;
      std::tie(threshold, size) = findWeightThreshold(percentile, wData, numAxonalArborLists, numPatches, nfp, nxp, nyp);


      const PVLayerLoc *targetLoc = this->targetLoc();
      const PVHalo *targetHalo = &targetLoc->halo;
      const int targetNx = targetLoc->nx;
      const int targetNy = targetLoc->ny;
      const int targetNf = targetLoc->nf;

      for (int arbor = 0; arbor < numAxonalArborLists; arbor++) {
         for (int kTargetRes = 0; kTargetRes < getNumPostNeurons(); kTargetRes++) {
            // Change restricted to extended post neuron
            int kTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);
            // get source layer's patch y stride
            int syp = postConnYPatchStride();
            int yPatchSize = postConnYPatchSize();
            // Iterate through y patch
            int nk = postConnXPatchSize() * postConnFPatchSize();
            int kernelIndex = postConnPatchToDataLUT(kTargetExt);

            const WeightType * const weightDataStart = postConnWDataHead(arbor, kernelIndex);

            for (int ky = 0; ky < yPatchSize; ky++) {
               const WeightType * const weightPtr = weightDataStart + ky * syp;

               // Don't re-sparsify something that's already been put thru the sparsfication grinder
               bool shouldSparsify = false;

               // Find the weight pointers for this nk sized patch
               // Find the weight pointers for this nk sized patch
               typename WeightMapType::iterator sparseWeightValuesNk = mValues.find(nk);
               typename IndexMapType::iterator sparseWeightIndexesNk = mIndexes.find(nk);

               if (mValues.find(nk) == mValues.end()) {
                  // Weight pointers don't exist for this sized nk. Allocate a map for this nk
                  mValues.insert(make_pair(nk, WeightPtrMapType()));
                  mIndexes.insert(make_pair(nk, WeightIndexMapType()));
                  // Get references
                  sparseWeightValuesNk = mValues.find(nk);
                  sparseWeightIndexesNk = mIndexes.find(nk);
                  shouldSparsify = true;
               } else if (sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
                  // This nk group exists, but no weight pointer.
                  shouldSparsify = true;
               }

               if (shouldSparsify) {
                  WeightListType sparseWeight;
                  IndexListType idx;

                  for (int k = 0; k < nk; k++) {
                     WeightType weight = weightPtr[k];
                     if (std::abs(weight) >= threshold) {
                        sparseWeight.push_back(weight);
                        idx.push_back(k);
                     }
                  }
                  
                  sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
                  sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
               }
            }
         }
      }
   }
};

} // Namespace PV

#endif // SPARSE_WEIGHT_H_
