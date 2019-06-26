#ifndef INTERACTIVECONTEXT_HPP_
#define INTERACTIVECONTEXT_HPP_

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>

namespace PV {

class InteractiveContext {
   public:
      InteractiveContext(std::map<std::string, std::string> args, std::string params);
      ~InteractiveContext();
      void   beginRun();
      double advanceRun(unsigned int steps);
      void   finishRun();
      void   getLayerActivity(const char *layerName, std::vector<float> *data);
      void   getLayerState(const char *layerName, std::vector<float> *data);
      void   setLayerState(const char *layerName, std::vector<float> *data);
      void   getLayerShape(const char *layerName, PVLayerLoc *loc);
      bool   isFinished();
      void   getProbeValues(const char *probeName, std::vector<double> *data);
      int    getMPIShape(int *rows, int *cols, int *batches);
      int    getMPILocation(int *row, int *col, int *batch);


   private:
      void   message(std::shared_ptr<BaseMessage const> message);
      int    mMPIRows, mMPICols, mMPIBatches;
      int    mRow, mCol, mBatch, mRank;

   protected:
      HyPerCol *mHC;
      PV_Init  *mPVI;
      int       mArgC;
      char    **mArgV;


};


} /* namespace PV */

#endif
