#include "PVPython.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <columns/buildandrun.hpp>
#include <columns/Messages.hpp>


namespace py = pybind11;

namespace PV {

InteractiveContext *newContext(py::dict args, std::string params) {
   std::vector<std::string> cliArgs;
   InteractiveContext *ic = (InteractiveContext*)calloc(sizeof(InteractiveContext), 1);

   cliArgs.push_back("pvpython");

   if (args.contains("OutputPath")) {
      cliArgs.push_back("-o");
      cliArgs.push_back(py::str(args["OutputPath"]));
   }
   if (args.contains("ParamsFile")) {
      cliArgs.push_back("-p");
      cliArgs.push_back(py::str(args["ParamsFile"]));
   }
   if (args.contains("LogFile")) {
      cliArgs.push_back("-l");
      cliArgs.push_back(py::str(args["LogFile"]));
   }
   if (args.contains("GPUDevices")) {
      cliArgs.push_back("-d");
      cliArgs.push_back(py::str(args["GPUDevices"]));
   }
   if (args.contains("RandomSeed")) {
      cliArgs.push_back("-s");
      cliArgs.push_back(py::str(args["RandomSeed"]));
   }
   if (args.contains("WorkingDirectory")) {
      cliArgs.push_back("-w");
      cliArgs.push_back(py::str(args["WorkingDirectory"]));
   }
   if (args.contains("Restart")) {
      if (args["Restart"].cast<bool>() == true) {
         cliArgs.push_back("-r");
      }
   }
   if (args.contains("CheckpointReadDirectory")) {
      cliArgs.push_back("-c");
      cliArgs.push_back(py::str(args["CheckpointReadDirectory"]));
   }
   if (args.contains("NumThreads")) {
      cliArgs.push_back("-t");
      cliArgs.push_back(py::str(args["NumThreads"]));
   }
   if (args.contains("BatchWidth")) {
      cliArgs.push_back("-batchwidth");
      cliArgs.push_back(py::str(args["BatchWidth"]));
   }
   if (args.contains("NumRows")) {
      cliArgs.push_back("-rows");
      cliArgs.push_back(py::str(args["NumRows"]));
   }
   if (args.contains("NumColumns")) {
      cliArgs.push_back("-columns");
      cliArgs.push_back(py::str(args["NumColumns"]));
   }
   if (args.contains("DryRun")) {
      if (args["DryRun"].cast<bool>() == true) {
         cliArgs.push_back("-n");
      }
   }
   if (args.contains("Shuffle")) {
      cliArgs.push_back("-shuffle");
      cliArgs.push_back(py::str(args["Shuffle"]));
   }

   ic->argc = cliArgs.size();
   ic->argv = (char**)calloc(ic->argc + 1, sizeof(char*));
   ic->argv[ic->argc] = NULL;
   for (int i = 0; i < ic->argc; i++) {
      ic->argv[i] = (char*)calloc(cliArgs[i].length()+1, sizeof(char));
      strcpy(ic->argv[i], cliArgs[i].c_str());
   }

   ic->initObj = new PV_Init(&ic->argc, &ic->argv, false);

   
   if (!params.empty()) {
      ic->initObj->setParamsBuffer(params.c_str(), params.length());
   }
   

   return ic;
}

void freeContext(InteractiveContext *ic) {
   delete(ic->hc);
   delete(ic->initObj);
   for (int i = 0; i < ic->argc; i++) {
      free(ic->argv[i]);
   }
   free(ic->argv);
   free(ic);
}

void beginRun(InteractiveContext *ic) {
   if (ic->initObj->isExtraProc()) {
      return;
   }
   PVParams *params = ic->initObj->getParams();
   if (params == NULL) {
      if (ic->initObj->getWorldRank() == 0) {
         char const *progName = ic->initObj->getProgramName();
         if (progName == NULL) {
            progName = "PetaVision";
         }
         ErrorLog().printf("%s was called without having set a params file\n", progName);
      }
      MPI_Barrier(ic->initObj->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   // Ignoring param sweep here, fix later

   ic->hc = new HyPerCol(ic->initObj);
   ic->hc->startRun();
}

double advanceRun(InteractiveContext *ic, unsigned int steps) {
   return ic->hc->multiStep(steps); 
}

void finishRun(InteractiveContext *ic) {
   ic->hc->finishRun();
}

py::array_t<float> getLayerActivity(InteractiveContext *ic, const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf;
   auto msg = std::make_shared<LayerGetActivityMessage>(layerName, &temp, &nx, &ny, &nf);
   ic->hc->externalMessage(msg);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nx, ny, nf}, false);
   return result;
}

} /* namespace PV */
