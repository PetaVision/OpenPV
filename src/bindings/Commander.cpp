#include <bindings/Commander.hpp>


namespace PV {


// Public

Commander::Commander(std::map<std::string, std::string> args, std::string params, void (*errFunc)(std::string const)) {
   mInteractions = new Interactions(args, params);
   mErrFunc = errFunc;
}

Commander::~Commander() {
   delete mInteractions;
}

// Private helper methods

void Commander::rootSend(const void *buf, int num, MPI_Datatype dtype) {
   FatalIf(getRank() != 0, "Cannot call rootSend from non-root processes.\n");
   for (int i = 1; i < getCommSize(); i++) {
      MPI_Send(buf, num, dtype, i, MPI_TAG, MPI_COMM_WORLD);
   }
}

void Commander::nonRootSend(const void *buf, int num, MPI_Datatype dtype) {
   FatalIf(getRank() == 0, "Cannot call nonRootSend from the root process.\n");
   MPI_Send(buf, num, dtype, 0, MPI_TAG, MPI_COMM_WORLD);
}

void Commander::nonRootRecv(void *buf, int num, MPI_Datatype dtype) {
   MPI_Status stat;
   FatalIf(getRank() == 0, "Cannot call nonRootRecv from the root process.\n");
   MPI_Recv(buf, num, dtype, 0, MPI_TAG, MPI_COMM_WORLD, &stat);
}

void Commander::rootSendCmdName(Command cmd, const char *name) {
   unsigned int len = strlen(name) + 1; /* account for null termination */
   rootSend(&cmd, 1, MPI_INT);
   rootSend(&len, 1, MPI_UNSIGNED);
   rootSend(name, len, MPI_CHAR);
}

std::string const Commander::nonRootRecvName() {
   unsigned int len;
   char *name;
   nonRootRecv(&len, 1, MPI_UNSIGNED);
   name = (char*)calloc(sizeof(char), len);
   nonRootRecv(name, len, MPI_CHAR);
   std::string ans(name);
   free(name);
   return ans;
}

int Commander::getRank() {
   return mInteractions->getMPILocation(NULL, NULL, NULL);
}

int Commander::getCommSize() {
   return mInteractions->getMPIShape(NULL, NULL, NULL);
}

int Commander::getRow() {
   int r;
   mInteractions->getMPILocation(&r, NULL, NULL);
   return r;
}

int Commander::getCol() {
   int c;
   mInteractions->getMPILocation(NULL, &c, NULL);
   return c;
}

int Commander::getBatch() {
   int b;
   mInteractions->getMPILocation(NULL, NULL, &b);
   return b;
}

// If an error callback function was provided, call it. Otherwise, throw an exception
void Commander::throwError(std::string const err) {
   std::string s = "<error on rank " + std::to_string(getRank()) + "> " + err;
   ErrorLog() << s << std::endl;
   if (mErrFunc != nullptr) {
      mErrFunc(s);
   }
   else {
      throw std::runtime_error(s.c_str());
   }
}


// Public methods for interacting with PetaVision

bool Commander::isRoot() {
   return getRank() == 0;
}

bool Commander::isFinished() {
   return mInteractions->isFinished();
}

double Commander::getLastCheckpointTime() {
   return mInteractions->lastCheckpointTime();
}

void Commander::waitForCommands() {
   bool finished = false;
   if (getRank() == 0) {
      throwError("waitForCommands can only be called from non-root processes.\n");
   }
   while (!finished) {
      Command cmd = CMD_NONE;
      nonRootRecv(&cmd, 1, MPI_INT);
      switch(cmd) {
         case CMD_BEGIN:
            if (mInteractions->begin() == Interactions::FAILURE) {
               throwError("waitForCommands: " + mInteractions->getError());
            }
            break;
         case CMD_FINISH:
            if (mInteractions->finish() == Interactions::FAILURE) {
               throwError("waitForCommands: " + mInteractions->getError());
            }
            finished = true;
            break;
         case CMD_ADVANCE:
            remoteAdvance();
            break;
         case CMD_GET_SPARSE_ACTIVITY:
            remoteGetLayerSparseActivity();
            break;
         case CMD_GET_ACTIVITY:
            remoteGetLayerData(BUF_A);
            break;
         case CMD_GET_STATE:
            remoteGetLayerData(BUF_V);
            break;
         case CMD_GET_PROBE_VALUES:
            remoteGetProbeValues();
            break;
         case CMD_SET_STATE:
            remoteSetLayerState();
            break;
         case CMD_SET_WEIGHTS:
            remoteSetConnectionWeights();
            break;
         case CMD_CHECKPOINT:
            if (mInteractions->checkpoint() == Interactions::FAILURE) {
               throwError("checkpoint: " + mInteractions->getError());
            }
            break;
         default:
            throwError("waitForCommands: unknown command received");
            break;
      }
   }
}

void Commander::checkpoint() {
   if (getRank() != 0) {
      throwError("checkpoint can only be called from the root process. "
            "Did you forget to call waitForCommands?");
   }
   const Command cmd = CMD_CHECKPOINT;
   rootSend(&cmd, 1, MPI_INT);
   if (mInteractions->checkpoint() == Interactions::FAILURE) {
      throwError("checkpoint: " + mInteractions->getError());
   }
}

void Commander::getLayerShape(const char *layerName, int *nb, int *ny, int *nx, int *nf) {
   if (getRank() != 0) {
      throwError("getLayerShape can only be called from the root process. "
            "Did you forget to call waitForCommands?");
   }
   PVLayerLoc loc;
   if (mInteractions->getLayerShape(layerName, &loc) == Interactions::FAILURE) {
      throwError("getLayerShape: " + mInteractions->getError());
   }

   if (nb != nullptr) {
      *nb = loc.nbatchGlobal;
   }
   if (ny != nullptr) {
      *ny = loc.nyGlobal;
   }
   if (nx != nullptr) {
      *nx = loc.nxGlobal;
   }
   if (nf != nullptr) {
      *nf = loc.nf;
   }
}

void Commander::getLayerSparseActivity(const char *layerName,
              std::vector<std::vector<std::pair<float, int>>> *data, int *ny, int *nx, int *nf) {
   if (getRank() != 0) {
      throwError("getLayerSparseActivity can only be called from the root process. "
            "Did you forget to call waitForCommands?");
   }

   PVLayerLoc loc;
   if (mInteractions->getLayerShape(layerName, &loc) == Interactions::FAILURE) {
      throwError("getLayerSparseActivity: " + mInteractions->getError());
   }

   if (ny != nullptr) {
      *ny = loc.nyGlobal;
   }
   if (nx != nullptr) {
      *nx = loc.nxGlobal;
   }
   if (nf != nullptr) {
      *nf = loc.nf;
   }

   std::vector<std::pair<float, int>> tempData;
   if (mInteractions->getLayerSparseActivity(layerName, &tempData) == Interactions::FAILURE) {
      throwError("getLayerSparseActivity: " + mInteractions->getError());
   }

   rootSendCmdName(CMD_GET_SPARSE_ACTIVITY, layerName);

   int batch, col, row, num;
   batch = loc.kb0;
   col   = loc.kx0;
   row   = loc.ky0;
   num   = tempData.size();

   data->clear();
   data->resize(loc.nbatchGlobal);
   MPI_Status stat;

   for (int r = 0; r < getCommSize(); r++) {
      if (r > 0) {
         MPI_Recv(&batch, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&col,   1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&row,   1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&num,   1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         tempData.resize(num);
         MPI_Recv(tempData.data(), num, MPI_FLOAT_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
      }
      // Modify our index to start from 0 for each batch
      // and correct for row / column offsets
      for (int b = batch; b < batch + loc.nbatch; b++) {
         int origin = b * (loc.nxGlobal*loc.nyGlobal*loc.nf);
         int offset = row * (loc.nxGlobal*loc.nf) + col * loc.nf;
         for (auto& v : tempData) {
            v.second -= origin;
            v.second += offset;
            data->at(b).push_back(v);
         }
      }
   }
}

void Commander::getLayerActivity(const char *layerName, std::vector<float> *data,
         int *nb, int *ny, int *nx, int *nf) {
   if (getRank() != 0) {
      throwError("getLayerActivity can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }
   getLayerData(layerName, data, nb, ny, nx, nf, BUF_A);
}

void Commander::getLayerState(const char *layerName, std::vector<float> *data,
         int *nb, int *ny, int *nx, int *nf) {
   if (getRank() != 0) {
      throwError("getLayerState can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }
   getLayerData(layerName, data, nb, ny, nx, nf, BUF_V);
}

void Commander::setLayerState(const char *layerName, std::vector<float> *data) {
   if (getRank() != 0) {
      throwError("setLayerState() can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }

   PVLayerLoc loc;
   if (mInteractions->getLayerShape(layerName, &loc) == Interactions::FAILURE) {
      throwError("setLayerState: " + mInteractions->getError());
   }

   unsigned int globalSize = loc.nxGlobal * loc.nyGlobal * loc.nf * loc.nbatchGlobal;
   unsigned int localSize  = loc.nx * loc.ny * loc.nf * loc.nbatch;
   int rows, cols, batches;
   int kb0 = loc.kb0;
   int kx0 = loc.kx0;
   int ky0 = loc.ky0;

   if(data->size() != globalSize) {
      throwError("setLayerState: vector has incorrect size. Got " + std::to_string(data->size())
            + ", expected " + std::to_string(globalSize));
   }

   rootSendCmdName(CMD_SET_STATE, layerName);

   MPI_Status stat;

   for (int r = 0; r < getCommSize(); r++) {
      if (r > 0) {
         MPI_Recv(&kb0, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&kx0, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&ky0, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
      }
      std::vector<float> slice;
      slice.resize(localSize);
      int dst = 0;
      for (int b = kb0; b < kb0 + loc.nbatch; b++) {
         for (int y = ky0; y < ky0 + loc.ny; y++) {
            for (int x = kx0; x < kx0 + loc.nx; x++) {
               for (int f = 0; f < loc.nf; f++) {
                  int src = b * (loc.nxGlobal*loc.nyGlobal*loc.nf) + y * (loc.nxGlobal*loc.nf) + x * loc.nf + f;
                  slice.at(dst++) = data->at(src);
               }
            }
         }
      }
      if (r > 0) {
         MPI_Send(slice.data(), slice.size(), MPI_FLOAT, r, MPI_TAG, MPI_COMM_WORLD);
      }
      else {
         if (mInteractions->setLayerState(layerName, &slice) == Interactions::FAILURE) {
            throwError("setLayerState: " + mInteractions->getError());
         }
      }
   }
}

void Commander::getProbeValues(const char *probeName, std::vector<double> *data) {
   if(getRank() != 0) {
      throwError("getProbeValues can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }
   std::vector<double> tempData;

   // Send the command early so MPI doesn't hang inside calcValues
   rootSendCmdName(CMD_GET_PROBE_VALUES, probeName);
   if (mInteractions->getProbeValues(probeName, &tempData) == Interactions::FAILURE) {
      throwError("getProbeValues: " + mInteractions->getError());
   }

   if (tempData.size() <= 0) {
      throwError("getProbeValues: no values found");
   }


   MPI_Status stat;
   int batches = 0;
   int batch   = getBatch();

   mInteractions->getMPIShape(NULL, NULL, &batches);
   data->resize(tempData.size() * batches);

   // This does redundant recvs when rows or cols > 1
   for (int r = 0; r < getCommSize(); r++) {
      if (r > 0) {
         MPI_Recv(&batch, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(tempData.data(), tempData.size(), MPI_DOUBLE, r, MPI_TAG, MPI_COMM_WORLD, &stat);
      }
      for (int i = 0; i < tempData.size(); i++) {
         int idx = batch * tempData.size() + i;
         data->at(idx) = tempData.at(i);
      }
   }
}

void Commander::getConnectionWeights(const char *connName, float **data,
                  int *nwp, int *nyp, int *nxp, int *nfp) {
   if(getRank() != 0) {
      throwError("getConnectionWeights can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }
   if (mInteractions->getConnectionPatchGeometry(connName, nwp, nyp, nxp, nfp) == Interactions::FAILURE) {
      throwError("getConnectionWeights: " + mInteractions->getError());
   }
   if (mInteractions->getConnectionWeights(connName, data) == Interactions::FAILURE) {
      throwError("getConnectionWeights: " + mInteractions->getError());
   }
}

void Commander::setConnectionWeights(const char *connName, std::vector<float> *data) {
   if(getRank() != 0) {
      throwError("setConnectionWeights can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }

   if (mInteractions->setConnectionWeights(connName, data) == Interactions::FAILURE) {
      throwError("setConnectionWeights: " + mInteractions->getError());
   }

   rootSendCmdName(CMD_SET_WEIGHTS, connName);
   rootSend(data->data(), data->size(), MPI_FLOAT);
}


void Commander::begin() {
   if (getRank() != 0) {
      throwError("begin can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }
   const Command cmd = CMD_BEGIN;
   rootSend(&cmd, 1, MPI_INT);

   if (mInteractions->begin() == Interactions::FAILURE) {
      throwError("begin: " + mInteractions->getError());
   }
}

void Commander::finish() {
   if(getRank() != 0) {
      throwError("finish can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }
   const Command cmd = CMD_FINISH;
   rootSend(&cmd, 1, MPI_INT);
   if (mInteractions->finish() == Interactions::FAILURE) {
      throwError("finish: " + mInteractions->getError());
   }
}

double Commander::advance(unsigned int steps) {
   if(getRank() != 0) {
      throwError("advance can only be called from the root process. "
         "Did you forget to call waitForCommands?");
   }
   double simTime;
   const Command cmd = CMD_ADVANCE;
   rootSend(&cmd,   1, MPI_INT);
   rootSend(&steps, 1, MPI_UNSIGNED);
   while (steps-- > 0) {
      if (mInteractions->step(&simTime) == Interactions::FAILURE) {
         throwError("advance: " + mInteractions->getError());
      }
   }
   return simTime;
}

void Commander::sendOk(int ok) {
   rootSend(&ok, 1, MPI_INT); 
}

int Commander::waitForOk() {
   int ok;
   nonRootRecv(&ok, 1, MPI_INT);
   return ok;
}

// The only difference between getLayerActivity and getLayerState is the buffer it
// fetches data from, so most of the implementation is here
void Commander::getLayerData(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf, Buffer b) {
   PVLayerLoc loc;
   if (mInteractions->getLayerShape(layerName, &loc) == Interactions::FAILURE) {
      throwError("getLayerData: " + mInteractions->getError());
   }

   unsigned int size = loc.nx * loc.ny * loc.nf * loc.nbatch;
   auto result = Interactions::SUCCESS;
   float **tempData = nullptr;
   switch(b) {
      case BUF_A:
         result = mInteractions->getLayerActivity(layerName, tempData);
         break;
      case BUF_V:
         result = mInteractions->getLayerState(layerName, tempData);
         break;
      default:
         break;
   }

   if (result == Interactions::FAILURE || tempData == nullptr) {
      throwError("getLayerData: " + mInteractions->getError());
   }

   // Send the command and arguments
   rootSendCmdName(b == BUF_A ? CMD_GET_ACTIVITY : CMD_GET_STATE, layerName);

   if (nx != nullptr) {
      *nx = loc.nxGlobal;
   }
   if (ny != nullptr) {
      *ny = loc.nyGlobal;
   }
   if (nf != nullptr) {
      *nf = loc.nf;
   }
   if (nb != nullptr) {
      *nb = loc.nbatchGlobal;
   }

   data->resize(loc.nxGlobal * loc.nyGlobal * loc.nf * loc.nbatchGlobal);

   int batch, col, row;
   batch = loc.kb0;
   col   = loc.kx0;
   row   = loc.ky0;

   // On the first loop, tempData has the root process' data. After that, it fetches from
   // other processes to assemble the rest
   MPI_Status stat;
   for (int r = 0; r < getCommSize(); r++) {
      if (r > 0) {
         MPI_Recv(&batch, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&col,   1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&row,   1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(*tempData, size, MPI_FLOAT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
      }
      int src = 0;
      for (int b = batch; b < batch + loc.nbatch; b++) {
         for (int y = row; y < row + loc.ny; y++) {
            for (int x = col; x < col + loc.nx; x++) {
               for (int f = 0; f < loc.nf; f++) {
                  int dst = b * (loc.nxGlobal*loc.nyGlobal*loc.nf) + y * (loc.nxGlobal*loc.nf) + x * loc.nf + f;
                  data->at(dst) = (*tempData)[src++]; 
               }
            }
         }
      }
   }
}


// Private commands that are only called by waitForCommand in reponse to MPI commands from root 

void Commander::remoteAdvance() {
   unsigned int steps;
   nonRootRecv(&steps, 1, MPI_UNSIGNED);
   while (steps-- > 0) {
      if (mInteractions->step(nullptr) == Interactions::FAILURE) {
         throwError("remoteAdvance: " + mInteractions->getError());
      }
   }
}

void Commander::remoteGetLayerSparseActivity() {
   std::string name = nonRootRecvName();
   PVLayerLoc loc;
   if (mInteractions->getLayerShape(name.c_str(), &loc) == Interactions::FAILURE) {
      throwError("remoteGetLayerSparseActivity: " + mInteractions->getError());
   }
   std::vector<std::pair<float, int>> data;
   if (mInteractions->getLayerSparseActivity(name.c_str(), &data) == Interactions::FAILURE) {
      throwError("remoteGetLayerSparseActivity: " + mInteractions->getError());
   }
   int size = data.size();
   nonRootSend(&loc.kb0, 1, MPI_INT);
   nonRootSend(&loc.kx0, 1, MPI_INT);
   nonRootSend(&loc.ky0, 1, MPI_INT);
   nonRootSend(&size,    1, MPI_INT);
   nonRootSend(data.data(), size, MPI_FLOAT_INT); 
}

void Commander::remoteGetLayerData(Buffer b) {
   std::string name = nonRootRecvName();
   PVLayerLoc loc;
   if (mInteractions->getLayerShape(name.c_str(), &loc) == Interactions::FAILURE) {
      throwError("remoteGetLayerData: " + mInteractions->getError());
   }

   auto result = Interactions::SUCCESS;
   float **data = nullptr;
   switch(b) {
      case BUF_A:
         result = mInteractions->getLayerActivity(name.c_str(), data);
         break;
      case BUF_V:
         result = mInteractions->getLayerState(name.c_str(), data);
         break;
      default:
         break;
   }

   if (result == Interactions::FAILURE || data == nullptr) {
      throwError("remoteGetLayerData: " + mInteractions->getError());
   }

   nonRootSend(&loc.kb0, 1, MPI_INT);
   nonRootSend(&loc.kx0, 1, MPI_INT);
   nonRootSend(&loc.ky0, 1, MPI_INT);
   nonRootSend(*data, loc.nbatch * loc.ny * loc.nx * loc.nf, MPI_FLOAT); 
}

void Commander::remoteSetLayerState() {
   PVLayerLoc loc;
   std::string name = nonRootRecvName();

   if (mInteractions->getLayerShape(name.c_str(), &loc) == Interactions::FAILURE) {
      throwError("remoteSetLayerState: " + mInteractions->getError());
   }
   nonRootSend(&loc.kb0, 1, MPI_INT);
   nonRootSend(&loc.kx0, 1, MPI_INT);
   nonRootSend(&loc.ky0, 1, MPI_INT);

   std::vector<float> data;
   int size = loc.nx * loc.ny * loc.nf * loc.nbatch;
   data.resize(size);
   nonRootRecv(data.data(), size, MPI_FLOAT);
   if (mInteractions->setLayerState(name.c_str(), &data) == Interactions::FAILURE) {
      throwError("remoteSetLayerState: " + mInteractions->getError());
   }
}

void Commander::remoteGetProbeValues() {
   std::string name = nonRootRecvName();
   int batch = getBatch();
   nonRootSend(&batch, 1, MPI_INT);

   std::vector<double> data;
   if (mInteractions->getProbeValues(name.c_str(), &data) == Interactions::FAILURE) {
      throwError("remoteGetProbeValues: " + mInteractions->getError());
   }
   nonRootSend(data.data(), data.size(), MPI_DOUBLE);
}

void Commander::remoteSetConnectionWeights() {
   std::string name = nonRootRecvName();
   int nwp = 0;
   int nyp = 0;
   int nxp = 0;
   int nfp = 0;
   if (mInteractions->getConnectionPatchGeometry(name.c_str(), &nwp, &nyp, &nxp, &nfp) == Interactions::FAILURE) {
      throwError("remoteSetConnectionWeights: " + mInteractions->getError());
   }
   std::vector<float> temp;
   temp.resize(nwp * nyp * nxp * nfp);
   nonRootRecv(temp.data(), temp.size(), MPI_FLOAT);
   if (mInteractions->setConnectionWeights(name.c_str(), &temp) == Interactions::FAILURE) {
      throwError("remoteSetConnectionWeights: " + mInteractions->getError());
   }
}


} /* namespace PV */
