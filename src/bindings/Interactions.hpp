#ifndef INTERACTIONS_HPP_
#define INTERACTIONS_HPP_

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>

namespace PV {

class Interactions {
   public:
      Interactions(std::map<std::string, std::string> args, std::string params);
      ~Interactions();

      enum Result {
         SUCCESS,
         FAILURE
      };

      Result beginRun();
      Result advanceRun(unsigned int steps, double *simTime);
      Result finishRun();
      Result getLayerActivity(const char *layerName, std::vector<float> *data);
      Result getLayerState(const char *layerName, std::vector<float> *data);
      Result setLayerState(const char *layerName, std::vector<float> const *data);
      Result getLayerShape(const char *layerName, PVLayerLoc *loc);
      Result getProbeValues(const char *probeName, std::vector<double> *data);
      Result getConnectionWeights(const char *connName, std::vector<float> *data);
      Result setConnectionWeights(const char *connName, std::vector<float> const *data);
      Result getConnectionPatchGeometry(const char *connName, int *nwp, int *nyp, int *nxp, int *nfp);
      bool   isFinished();
      int    getMPIShape(int *rows, int *cols, int *batches);
      int    getMPILocation(int *row, int *col, int *batch);
      std::string const getError();

   private:
      Response::Status interact(std::shared_ptr<InteractionMessage const> message);
      void   clearError();
      void   error(std::string const err);
      Result checkError(std::shared_ptr<InteractionMessage const> message, Response::Status status,
            std::string const funcName, std::string const objName);
      int    mMPIRows, mMPICols, mMPIBatches;
      int    mRow, mCol, mBatch, mRank;
      std::string mErrMsg = "";

   protected:
      HyPerCol *mHC;
      PV_Init  *mPVI;
      int       mArgC;
      char    **mArgV;


};


} /* namespace PV */

#endif
