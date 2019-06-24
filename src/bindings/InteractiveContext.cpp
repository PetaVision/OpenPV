#include <bindings/InteractiveContext.hpp>
#include <columns/Messages.hpp>


namespace PV {

void InteractiveContext::message(std::shared_ptr<BaseMessage const> message) {
   mHC->externalMessage(message);
}

// These wrap some of the repetitive loop writing in a simpler call
void InteractiveContext::rootSend(const void *buf, int num, MPI_Datatype dtype) {
   FatalIf(getMPIRank() != 0, "Cannot call InteractiveContext::rootSend from non root process.\n");
   for (int i = 1; i < getMPICommSize(); i++) {
      MPI_Send(buf, num, dtype, i, IC_MPI_TAG, MPI_COMM_WORLD);
   }
}

void InteractiveContext::nonRootSend(const void *buf, int num, MPI_Datatype dtype) {
   FatalIf(getMPIRank() == 0, "Cannot call InteractiveContext::nonRootSend from root process.\n");
   MPI_Send(buf, num, dtype, 0, IC_MPI_TAG, MPI_COMM_WORLD);
}

void InteractiveContext::nonRootRecv(void *buf, int num, MPI_Datatype dtype) {
   MPI_Status stat;
   FatalIf(getMPIRank() == 0, "Cannot call InteractiveContext::nonRootSend from root process.\n");
   MPI_Recv(buf, num, dtype, 0, IC_MPI_TAG, MPI_COMM_WORLD, &stat);
}


InteractiveContext::InteractiveContext(std::map<std::string, std::string> args, std::string params) {
   std::vector<std::string> cliArgs;

   cliArgs.push_back("OpenPV");

   if (args.find("OutputPath") != args.end()) {
      cliArgs.push_back("-o");
      cliArgs.push_back(args["OutputPath"]);
   }
   if (args.find("ParamsFile") != args.end()) {
      cliArgs.push_back("-p");
      cliArgs.push_back(args["ParamsFile"]);
   }
   if (args.find("LogFile") != args.end()) {
      cliArgs.push_back("-l");
      cliArgs.push_back(args["LogFile"]);
   }
   if (args.find("GPUDevices") != args.end()) {
      cliArgs.push_back("-d");
      cliArgs.push_back(args["GPUDevices"]);
   }
   if (args.find("RandomSeed") != args.end()) {
      cliArgs.push_back("-s");
      cliArgs.push_back(args["RandomSeed"]);
   }
   if (args.find("WorkingDirectory") != args.end()) {
      cliArgs.push_back("-w");
      cliArgs.push_back(args["WorkingDirectory"]);
   }
   if (args.find("Restart") != args.end() && args["Restart"] == "True") {
      cliArgs.push_back("-r");
   }
   if (args.find("CheckpointReadDirectory") != args.end()) {
      cliArgs.push_back("-c");
      cliArgs.push_back(args["CheckpointReadDirectory"]);
   }
   if (args.find("NumThreads") != args.end()) {
      cliArgs.push_back("-t");
      cliArgs.push_back(args["NumThreads"]);
   }
   if (args.find("BatchWidth") != args.end()) {
      cliArgs.push_back("-batchwidth");
      cliArgs.push_back(args["BatchWidth"]);
   }
   if (args.find("NumRows") != args.end()) {
      cliArgs.push_back("-rows");
      cliArgs.push_back(args["NumRows"]);
   }
   if (args.find("NumColumns") != args.end()) {
      cliArgs.push_back("-columns");
      cliArgs.push_back(args["NumColumns"]);
   }
   if (args.find("DryRun") != args.end() && args["DryRun"] == "True") {
      cliArgs.push_back("-n");
   }
   if (args.find("Shuffle") != args.end()) {
      cliArgs.push_back("-shuffle");
      cliArgs.push_back(args["Shuffle"]);
   }

   // Build argc and argv for PV_Init
   mArgC = cliArgs.size();
   mArgV = (char**)calloc(mArgC + 1, sizeof(char*));
   mArgV[mArgC] = NULL;
   for (int i = 0; i < mArgC; i++) {
      mArgV[i] = (char*)calloc(cliArgs[i].length()+1, sizeof(char));
      strcpy(mArgV[i], cliArgs[i].c_str());
   }

   mPVI = new PV_Init(&mArgC, &mArgV, false);

   // Read params from a string instead of a file
   if (!params.empty()) {
      mPVI->setParamsBuffer(params.c_str(), params.length());
   }
}

InteractiveContext::~InteractiveContext() {
   delete(mHC);
   delete(mPVI);
   for (int i = 0; i < mArgC; i++) {
      free(mArgV[i]);
   }
   free(mArgV);
}

void InteractiveContext::beginRun() {
   if (mPVI->isExtraProc()) {
      return; //TODO: Some sort of feedback?
   }
   PVParams *params = mPVI->getParams();
   if (params == NULL) {
      if (mPVI->getWorldRank() == 0) {
         char const *progName = mPVI->getProgramName();
         if (progName == NULL) {
            progName = "PetaVision";
         }
         ErrorLog().printf("%s was called without having set a params file\n", progName);
      }
      MPI_Barrier(mPVI->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   // TODO: Ignoring param sweep here, fix later

   mHC = new HyPerCol(mPVI);
   mHC->startRun();
}

void InteractiveContext::handleMPI() {
   if (getMPIRank() == 0) {
      return;
   }
   while (!isFinished()) {
      Command cmd = CMD_NONE;
      nonRootRecv(&cmd, 1, MPI_INT);
      switch(cmd) {
         case CMD_ADVANCE_RUN:
            {
               unsigned int steps;
               nonRootRecv(&steps, 1, MPI_UNSIGNED);
               advanceRun(steps);
            }
            break;
         case CMD_GET_ACTIVITY:
            {
               unsigned int len;
               char *layerName;
               nonRootRecv(&len, 1, MPI_UNSIGNED);
               layerName = (char*)calloc(sizeof(char), len);
               nonRootRecv(layerName, len, MPI_CHAR);
               remoteGetLayerActivity(layerName);
               free(layerName);
            }
            break;
         default:
            break;
      }
   }
}

double InteractiveContext::advanceRun(unsigned int steps) {
   if (getMPIRank() == 0) {
      const Command cmd = CMD_ADVANCE_RUN;
      rootSend(&cmd,   1, MPI_INT);
      rootSend(&steps, 1, MPI_UNSIGNED);
   }
   return mHC->multiStep(steps); 
}

void InteractiveContext::finishRun() {
   mHC->finishRun();
}

void InteractiveContext::remoteGetLayerActivity(const char *layerName) {
   std::vector<float> data;
   PVLayerLoc loc;

   getLayerShape(layerName, &loc);
   message(std::make_shared<LayerGetActivityMessage>(layerName, &data));

   nonRootSend(&loc.kb0, 1, MPI_INT);
   nonRootSend(&loc.kx0, 1, MPI_INT);
   nonRootSend(&loc.ky0, 1, MPI_INT);
   nonRootSend(data.data(), data.size(), MPI_FLOAT); 
}

void InteractiveContext::getLayerActivity(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf, int *nb) {
   getLayerData(layerName, data, nx, ny, nf, nb, BUF_A);
}

void InteractiveContext::getLayerState(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf, int *nb) {
   getLayerData(layerName, data, nx, ny, nf, nb, BUF_V);
}

void InteractiveContext::getLayerData(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf, int *nb, Buffer b) {

   FatalIf(getMPIRank() != 0, "InteractiveContext::getLayerData can only be called from root process. "
         "Did you forget to call handleMPI()?");

   MPI_Status stat;
   std::vector<float> tempData;
   const Command cmd = CMD_GET_ACTIVITY;
   unsigned int len = strlen(layerName) + 1;
   unsigned int size;
   int batch, col, row;

   // Send the command and arguments
   rootSend(&cmd, 1, MPI_INT);
   rootSend(&len, 1, MPI_UNSIGNED);
   rootSend(layerName, len, MPI_CHAR);

   PVLayerLoc loc;
   getLayerShape(layerName, &loc);
   size = loc.nx * loc.ny * loc.nf * loc.nbatch;
   if (b == BUF_A) {
      message(std::make_shared<LayerGetActivityMessage>(layerName, &tempData));
   }
   else {
      message(std::make_shared<LayerGetInternalStateMessage>(layerName, &tempData));
   }

   FatalIf(size != tempData.size(),
         "getLayerShape returned a different size than LayerGetData");

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
   for (int r = 0; r < getMPICommSize(); r++) {
      if (r > 0) {
         MPI_Recv(&batch, 1, MPI_INT, r, IC_MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&col,   1, MPI_INT, r, IC_MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(&row,   1, MPI_INT, r, IC_MPI_TAG, MPI_COMM_WORLD, &stat);
         MPI_Recv(tempData.data(), size, MPI_FLOAT, r, IC_MPI_TAG, MPI_COMM_WORLD, &stat);
      }
      int src = 0;
      for (int b = batch; b < batch + loc.nbatch; b++) {
         for (int y = row; y < row + loc.ny; y++) {
            for (int x = col; x < col + loc.nx; x++) {
               for (int f = 0; f < loc.nf; f++) {
                  int dst = b * (loc.nx*loc.ny*loc.nf) + y * (loc.nx*loc.nf) + x * loc.nf + f;
                  data->at(dst) = tempData.at(src++); 
               }
            }
         }
      }
   }
}

void InteractiveContext::setLayerState(const char *layerName, std::vector<float> *data) {
   // TODO: MPI
   message(std::make_shared<LayerSetInternalStateMessage>(layerName, data));
}

void InteractiveContext::getLayerShape(const char *layerName, PVLayerLoc *loc) {
   message(std::make_shared<LayerGetShapeMessage>(layerName, loc));
}

bool InteractiveContext::isFinished() {
   int fin = mHC->simulationTime() >= mHC->getStopTime() - mHC->getDeltaTime() / 2.0;
   int maxval = 0;
   // This needs to be communicated to non root processes
   MPI_Reduce(&fin, &maxval, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
   return maxval;
}

void InteractiveContext::getEnergy(const char *probeName, std::vector<double> *data) {
   message(std::make_shared<ColumnEnergyProbeGetEnergyMessage>(probeName, data));
}

int InteractiveContext::getMPIRank() {
   return mHC->getCommunicator()->globalCommRank();
}

int InteractiveContext::getMPICommSize() {
   return mHC->getCommunicator()->globalCommSize();
}

} /* namespace PV */
