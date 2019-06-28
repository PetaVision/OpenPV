#include <bindings/Commander.hpp>


namespace PV {


// Public

Commander::Commander(std::map<std::string, std::string> args, std::string params, void (*errFunc)(std::string const)) {
   mIC = new Interactions(args, params);
   mErrFunc = errFunc;
}

Commander::~Commander() {
   delete mIC;
}

bool Commander::isRoot() {
   return getRank() == 0;
}

bool Commander::isFinished() {
   return mIC->isFinished();
}

void Commander::waitForCommands() {
   bool finished = false;
   if (getRank() == 0) {
      mErrFunc("waitForCommands can only be called from non-root processes.\n");
      return;
   }
   while (!finished) {
      Command cmd = CMD_NONE;
      nonRootRecv(&cmd, 1, MPI_INT);
      switch(cmd) {
         case CMD_BEGIN_RUN:
            mIC->beginRun();
            break;
         case CMD_FINISH_RUN:
            mIC->finishRun();
            finished = true;
            break;
         case CMD_ADVANCE_RUN:
            remoteAdvanceRun();
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
         default:
            break;
      }
   }
}

void Commander::getLayerActivity(const char *layerName, std::vector<float> *data,
         int *nb, int *ny, int *nx, int *nf) {
   if(getRank() != 0) {
      mErrFunc("getLayerActivity only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   getLayerData(layerName, data, nb, ny, nx, nf, BUF_A);
}

void Commander::getLayerState(const char *layerName, std::vector<float> *data,
         int *nb, int *ny, int *nx, int *nf) {
   if(getRank() != 0) {
      mErrFunc("getLayerState can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   getLayerData(layerName, data, nb, ny, nx, nf, BUF_V);
}

void Commander::setLayerState(const char *layerName, std::vector<float> *data) {
   if (getRank() != 0) {
      mErrFunc("setLayerState() can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   const Command cmd = CMD_SET_STATE;
   unsigned int len = strlen(layerName) + 1;
   rootSend(&cmd, 1, MPI_INT);
   rootSend(&len, 1, MPI_UNSIGNED);
   rootSend(layerName, len, MPI_CHAR);

   MPI_Status stat;
   PVLayerLoc loc;
   if (mIC->getLayerShape(layerName, &loc) == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
   }

   unsigned int globalSize = loc.nxGlobal * loc.nyGlobal * loc.nf * loc.nbatchGlobal;
   unsigned int localSize  = loc.nx * loc.ny * loc.nf * loc.nbatch;
   int rows, cols, batches;
   int kb0 = loc.kb0;
   int kx0 = loc.kx0;
   int ky0 = loc.ky0;

   if(data->size() != globalSize) {
      mErrFunc("setLayerState() vector has incorrect size");
      return;
   }

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
         if (mIC->setLayerState(layerName, &slice) == Interactions::FAILURE) {
            mErrFunc(mIC->getError());
         }
      }
   }
}

void Commander::getProbeValues(const char *probeName, std::vector<double> *data) {
   if(getRank() != 0) {
      mErrFunc("getProbeValues can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   const Command cmd = CMD_GET_PROBE_VALUES;
   unsigned int len = strlen(probeName) + 1;
   rootSend(&cmd, 1, MPI_INT);
   rootSend(&len, 1, MPI_UNSIGNED);
   rootSend(probeName, len, MPI_CHAR);

   MPI_Status stat;
   std::vector<double> tempData;
   int batches;
   int batch = getBatch();

   mIC->getMPIShape(NULL, NULL, &batches);

   // This should probably happen before the MPI command is sent out
   if (mIC->getProbeValues(probeName, &tempData) == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return;
   }

   len = tempData.size();
   data->resize(len * batches);

   // This does redundant recvs when rows or cols > 1
   for (int r = 0; r < getCommSize(); r++) {
      if (r > 0) {
         MPI_Recv(&batch, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(tempData.data(), len, MPI_DOUBLE, r, MPI_TAG, MPI_COMM_WORLD, &stat);
      }
      for (int i = 0; i < tempData.size(); i++) {
         int idx = batch * tempData.size() + i;
         data->at(idx) = tempData.at(i);
      }
   }
}

void Commander::getConnectionWeights(const char *connName, std::vector<float> *data,
                  int *nwp, int *nyp, int *nxp, int *nfp) {
   if(getRank() != 0) {
      mErrFunc("getConnectionWweights can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   // TODO: Verify that this does not need any MPI interaction
   if (mIC->getConnectionPatchGeometry(connName, nwp, nyp, nxp, nfp) == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return;
   }
   if (mIC->getConnectionWeights(connName, data) == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return;
   }
}

void Commander::setConnectionWeights(const char *connName, std::vector<float> *data) {
   if(getRank() != 0) {
      mErrFunc("setConnectionWweights can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   // TODO: Verify that this does not need any MPI interaction
   if (mIC->setConnectionWeights(connName, data) == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return;
   }
}


void Commander::beginRun() {
   if (getRank() != 0) {
      mErrFunc("beginRun can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   const Command cmd = CMD_BEGIN_RUN;
   rootSend(&cmd,   1, MPI_INT);
   if (mIC->beginRun() == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return;
   }
}

void Commander::finishRun() {
   if(getRank() != 0) {
      mErrFunc("finishRun can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return;
   }
   const Command cmd = CMD_FINISH_RUN;
   rootSend(&cmd,   1, MPI_INT);
   if (mIC->finishRun() == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return;
   }
}

double Commander::advanceRun(unsigned int steps) {
   if(getRank() != 0) {
      mErrFunc("advanceRun can only be called from the root process. "
         "Did you forget to call waitForCommands?");
      return 0.0;
   }
   double simTime;
   const Command cmd = CMD_ADVANCE_RUN;
   rootSend(&cmd,   1, MPI_INT);
   rootSend(&steps, 1, MPI_UNSIGNED);
   if (mIC->advanceRun(steps, &simTime) == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return 0.0;
   }
   return simTime;
}

// Private


// These wrap some of the repetitive loop writing in a simpler call
void Commander::rootSend(const void *buf, int num, MPI_Datatype dtype) {
   FatalIf(getRank() != 0, "Cannot call rootSend() from non-root processes.\n");
   for (int i = 1; i < getCommSize(); i++) {
      MPI_Send(buf, num, dtype, i, MPI_TAG, MPI_COMM_WORLD);
   }
}

void Commander::nonRootSend(const void *buf, int num, MPI_Datatype dtype) {
   FatalIf(getRank() == 0, "Cannot call nonRootSend() from the root process.\n");
   MPI_Send(buf, num, dtype, 0, MPI_TAG, MPI_COMM_WORLD);
}

void Commander::nonRootRecv(void *buf, int num, MPI_Datatype dtype) {
   MPI_Status stat;
   FatalIf(getRank() == 0, "Cannot call nonRootSend() from the root process.\n");
   MPI_Recv(buf, num, dtype, 0, MPI_TAG, MPI_COMM_WORLD, &stat);
}

int Commander::getRank() {
   return mIC->getMPILocation(NULL, NULL, NULL);
}

int Commander::getCommSize() {
   return mIC->getMPIShape(NULL, NULL, NULL);
}

int Commander::getRow() {
   int r;
   mIC->getMPILocation(&r, NULL, NULL);
   return r;
}

int Commander::getCol() {
   int c;
   mIC->getMPILocation(NULL, &c, NULL);
   return c;
}

int Commander::getBatch() {
   int b;
   mIC->getMPILocation(NULL, NULL, &b);
   return b;
}

void Commander::getLayerData(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf, Buffer b) {
   Command cmd = CMD_NONE;
   unsigned int len = strlen(layerName) + 1;
   switch(b) {
      case BUF_A:
         cmd = CMD_GET_ACTIVITY;
         break;
      case BUF_V:
         cmd = CMD_GET_STATE;
         break;
      default:
         break;
   }

   // Send the command and arguments
   rootSend(&cmd, 1, MPI_INT);
   rootSend(&len, 1, MPI_UNSIGNED);
   rootSend(layerName, len, MPI_CHAR);

   MPI_Status stat;
   std::vector<float> tempData;
   PVLayerLoc loc;
   unsigned int size;
   int batch, col, row;


   mIC->getLayerShape(layerName, &loc);
   size = loc.nx * loc.ny * loc.nf * loc.nbatch;
   auto result = Interactions::SUCCESS;
   switch(b) {
      case BUF_A:
         result = mIC->getLayerActivity(layerName, &tempData);
         break;
      case BUF_V:
         result = mIC->getLayerState(layerName, &tempData);
         break;
      default:
         break;
   }

   if (result == Interactions::FAILURE) {
      mErrFunc(mIC->getError());
      return;
   }

   if(size != tempData.size()) {
      mErrFunc("getLayerShape returned an unexpected value.");
      return;
   }

   *nx = loc.nxGlobal;
   *ny = loc.nyGlobal;
   *nf = loc.nf;
   *nb = loc.nbatchGlobal;

   data->resize((*nx) * (*ny) * (*nf) * (*nb));

   batch = loc.kb0;
   col   = loc.kx0;
   row   = loc.ky0;

   // On the first loop, tempData has the root process' data. After that, it fetches from
   // other processes to assemble the rest
   for (int r = 0; r < getCommSize(); r++) {
      if (r > 0) {
         MPI_Recv(&batch, 1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&col,   1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&row,   1, MPI_INT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(tempData.data(), size, MPI_FLOAT, r, MPI_TAG, MPI_COMM_WORLD, &stat);
      }
      int src = 0;
      for (int b = batch; b < batch + loc.nbatch; b++) {
         for (int y = row; y < row + loc.ny; y++) {
            for (int x = col; x < col + loc.nx; x++) {
               for (int f = 0; f < loc.nf; f++) {
                  int dst = b * (loc.nxGlobal*loc.nyGlobal*loc.nf) + y * (loc.nxGlobal*loc.nf) + x * loc.nf + f;
                  data->at(dst) = tempData.at(src++); 
               }
            }
         }
      }
   }
}


// These commands are only called by waitForCommand in reponse to MPI commands from root 

void Commander::remoteAdvanceRun() {
   unsigned int steps;
   nonRootRecv(&steps, 1, MPI_UNSIGNED);
   mIC->advanceRun(steps, nullptr);
}

void Commander::remoteGetLayerData(Buffer b) {
   std::vector<float> data;
   PVLayerLoc loc;
   unsigned int len;
   char *layerName;

   nonRootRecv(&len, 1, MPI_UNSIGNED);
   layerName = (char*)calloc(sizeof(char), len);
   nonRootRecv(layerName, len, MPI_CHAR);

   mIC->getLayerShape(layerName, &loc);
   switch(b) {
      case BUF_A:
         mIC->getLayerActivity(layerName, &data);
         break;
      case BUF_V:
         mIC->getLayerState(layerName, &data);
         break;
      default:
         break;
   }

   nonRootSend(&loc.kb0, 1, MPI_INT);
   nonRootSend(&loc.kx0, 1, MPI_INT);
   nonRootSend(&loc.ky0, 1, MPI_INT);
   nonRootSend(data.data(), data.size(), MPI_FLOAT); 

   free(layerName);
}

void Commander::remoteSetLayerState() {
   std::vector<float> data;
   PVLayerLoc loc;
   unsigned int len;
   char *layerName;

   nonRootRecv(&len, 1, MPI_UNSIGNED);
   layerName = (char*)calloc(sizeof(char), len);
   nonRootRecv(layerName, len, MPI_CHAR);
   mIC->getLayerShape(layerName, &loc);

   len = loc.nx * loc.ny * loc.nf * loc.nbatch;
   data.resize(len);

   nonRootSend(&loc.kb0, 1, MPI_INT);
   nonRootSend(&loc.kx0, 1, MPI_INT);
   nonRootSend(&loc.ky0, 1, MPI_INT);
   nonRootRecv(data.data(), len, MPI_FLOAT);
   mIC->setLayerState(layerName, &data);
   
   free(layerName);
}

void Commander::remoteGetProbeValues() {
   std::vector<double> data;
   unsigned int len;
   char *probeName;

   nonRootRecv(&len, 1, MPI_UNSIGNED);
   probeName = (char*)calloc(sizeof(char), len);
   nonRootRecv(probeName, len, MPI_CHAR);

   int batch = getBatch();
   nonRootSend(&batch, 1, MPI_INT);

   mIC->getProbeValues(probeName, &data);
   nonRootSend(data.data(), data.size(), MPI_DOUBLE);

   free(probeName);
}


} /* namespace PV */
