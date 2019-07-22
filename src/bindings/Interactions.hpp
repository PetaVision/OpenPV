#ifndef INTERACTIONS_HPP_
#define INTERACTIONS_HPP_

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>

namespace PV {

/*
   The Interactions class provides an interface between the HyPerCol (and the objects it owns)
   and an external tool like Python or Matlab. This allows running PetaVision in an interactive
   context instead of using buildandrun. The methods here are not MPI aware, and only get / set
   values on the current process.
*/

class Interactions {
   public:
      /*
         args:
            A string to string map of command line arguments. The keys it accepts are:
               OutputPath
               ParamsFile
               LogFile
               GPUDevices
               RandomSeed
               WorkingDirectory
               Restart
               CheckpointReadDirectory
               NumThreads
               BatchWidth
               NumRows
               NumColumns
               DryRun
               Shuffle
         params:
            The contents of the params file to run, as a string. If this string is empty,
            it uses the file specified as ParamsFile in args.
      */
      Interactions(std::map<std::string, std::string> args, std::string params);
      ~Interactions();

      enum Result {
         SUCCESS,
         FAILURE
      };

      // Construct the HyPerCol and start a run
      Result begin();
      // Advance the run by one timestep and sets the value of simTime
      Result step(double *simTime);
      // Calls HyPerCol.finishRun()
      Result finish();
      // Fills the data vector with the restricted contents of the named layer's Activity buffer
      Result getLayerSparseActivity(const char *layerName, std::vector<std::pair<float, int>> *data);
      // Fills the data vector with the restricted contents of the named layer's Activity buffer
      Result getLayerActivity(const char *layerName, std::vector<float> *data);
      // Fills the data vector with the contents of the named layer's InternalState buffer
      Result getLayerState(const char *layerName, std::vector<float> *data);
      // Overwrites the contents of the layer's InternalState buffer with the contents of data
      Result setLayerState(const char *layerName, std::vector<float> const *data);
      // Sets loc to be a copy of the named layer's PVLayerLoc
      Result getLayerShape(const char *layerName, PVLayerLoc *loc);
      // Fills data with the results of calling getValues() on the named probe
      Result getProbeValues(const char *probeName, std::vector<double> *data);
      // Fills data with the patch data for the named connection's preweights
      Result getConnectionWeights(const char *connName, std::vector<float> *data);
      // Sets the preweight patch data for the named connection to data, and updates the postweights and GPU
      Result setConnectionWeights(const char *connName, std::vector<float> const *data);
      // Fetches the patch dimensions for the named connection
      Result getConnectionPatchGeometry(const char *connName, int *nwp, int *nyp, int *nxp, int *nfp);
      // Returns true if simTime >= stopTime
      bool   isFinished();
      // Fills rows / cols / batches with the appropriate values and returns the total number of ranks
      int    getMPIShape(int *rows, int *cols, int *batches);
      // Fills row / col / batch with the appropriate values and returns the rank of this process
      int    getMPILocation(int *row, int *col, int *batch);
      // Returns an empty string or the reason one of the above methods returned FAILURE
      std::string const getError();

   private:
      // Sends the HyPerCol a message that corresponds with one of the public methods
      Response::Status interact(std::shared_ptr<InteractionMessage const> message);
      // Internal error handling methods
      void   clearError();
      void   error(std::string const err);
      Result checkError(std::shared_ptr<InteractionMessage const> message, Response::Status status,
            std::string const funcName, std::string const objName);
      std::string mErrMsg = "";

      // Cache the MPI info for quick retrieval
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
