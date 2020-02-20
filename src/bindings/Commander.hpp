#ifndef COMMANDER_HPP_
#define COMMANDER_HPP_


#include <mpi.h>
#include <bindings/Interactions.hpp>

namespace PV {

/*
   The Commander class is an MPI aware interface for Interactions that assumes all 
   control of PetaVision will be coming from the root process. The Commander allows
   using PetaVision in an interactive context from a single location without the user
   having to be aware of the distribution of MPI processes. Aside from the constructor,
   the only method that runs on non-root processes is waitForCommands.
*/

class Commander {
   public:
      /*
         args, params:
            See Interactions.hpp
         errFunc: 
            A callback function that will be used when an error is detected. If errFunc
            is set to nullptr, errors will throw an std::runtime_error exception instead.
      */
      Commander(std::map<std::string, std::string> args, std::string params, void (*errFunc)(std::string const));
      ~Commander();

      static const int MPI_TAG = 666;

   protected:
      Interactions *mInteractions = nullptr;

   private:
      enum Command {
         CMD_NONE,
         CMD_BEGIN,
         CMD_ADVANCE,
         CMD_FINISH,
         CMD_GET_SPARSE_ACTIVITY,
         CMD_GET_ACTIVITY,
         CMD_GET_STATE,
         CMD_SET_STATE,
         CMD_GET_PROBE_VALUES,
         CMD_SET_WEIGHTS
      };

      enum Buffer {
         BUF_A,
         BUF_V
      };

      // Helper methods to reduce the amount of redundant MPI code
      void   rootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootRecv(void *buf, int num, MPI_Datatype dtype);
      void   rootSendCmdName(Command cmd, const char *name);
      std::string const nonRootRecvName();
      // Throws an exception or calls the callback error function, if specified
      void   throwError(std::string const err);
      void   (*mErrFunc)(std::string const);


   public:
      // Get MPI info
      int    getRank();
      int    getCommSize();
      int    getRow();
      int    getCol();
      int    getBatch();
      // The only MPI info the user must be aware of is if they are the root process or not
      bool   isRoot();
      // Returns true when simTime >= stopTime
      bool   isFinished();
      // On non-root processes, wait for the root process to send a message indicating
      // what methods were called. Loops until finish is called on the root process.
      void   waitForCommands();
      // Return the dimensions of a layer
      void   getLayerShape(const char *layerName, int *nb, int *ny, int *nx, int *nf);
      // Fill the data vector with index value pairs of non-zero Activity 
      // and returns the layer's shape to allow reshaping into the proper dimensions.
      // The outer std::vector is batch index.
      void   getLayerSparseActivity(const char *layerName,
              std::vector<std::vector<std::pair<float, int>>> *data, int *ny, int *nx, int *nf);
      // Fill the data vector with the contents of the named layer's restricted Activity 
      // buffer and returns the layer's shape to allow reshaping into the proper dimensions
      void   getLayerActivity(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf);
      // Fill the data vector with the contents of the named layer's InternalState 
      // buffer and returns the layer's shape to allow reshaping into the proper dimensions
      void   getLayerState(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf);
      // Sets the InternalState buffer of the named layer to the contents of the data
      // vector, which should be the same size as the one returned by getLayerState
      void   setLayerState(const char *layerName, std::vector<float> *data);
      // Fills data with the vector of values returned by calling getValues on the named probe
      void   getProbeValues(const char *probeName, std::vector<double> *data);
      // Fills data with the patch data for the named connection's preweights and returns the
      // connection's patch geometry to allow reshaping
      void   getConnectionWeights(const char *connName, std::vector<float> *data,
                  int *nwp, int *nyp, int *nxp, int *nfp);
      // Sets the preweight patch data for the named connection to data, and updates the
      // postweights and GPU
      void   setConnectionWeights(const char *connName, std::vector<float> *data);
      // Creates the HyPerCol and starts the run
      void   begin();
      // Call before exiting to properly clean up the end of a run
      void   finish();
      // Advances the run a specified number of timesteps. Returns simTime.
      double advance(unsigned int steps);

      void sendOk(int ok);
      int waitForOk(); 

   private:
      // Used by getLayerActivity and getLayerState to avoid code duplication
      void   getLayerData(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf, Buffer b);

      // These are called by waitForCommands to perform the appropriate actions on
      // non-root processes
      void   remoteAdvance();
      void   remoteGetLayerSparseActivity();
      void   remoteGetLayerData(Buffer b);
      void   remoteSetLayerState();
      void   remoteGetProbeValues();
      void   remoteSetConnectionWeights();
};


} /* namespace PV */




#endif
