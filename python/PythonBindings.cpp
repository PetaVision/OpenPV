#include "PythonBindings.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <thread>
#include <chrono>


namespace py = pybind11;

namespace PV {

PythonBindings::PythonBindings(py::dict args, std::string params) {
   int np = 1;
   for (auto ent : args) {
      args[ent.first] = py::str(ent.second);
      std::string first = py::cast<std::string>(py::str(ent.first));
      if (first == "BatchWidth"
            || first == "NumRows"
            || first == "NumCols") {
         std::string second = py::cast<std::string>(py::str(ent.second));
         np *= std::stoi(second);
      }
   }
   mCmd = new Commander(py::cast<std::map<std::string, std::string>>(args), params, &err);
   py::print("Rank " + std::to_string(mCmd->getRank()+1) + " of " + std::to_string(mCmd->getCommSize()) + " initialized");
   if (mCmd->getRank() == 0) {
      if (mCmd->getCommSize() != np) {
         std::this_thread::sleep_for(std::chrono::seconds(1));
         py::print("np = " + std::to_string(mCmd->getCommSize())
               + ", but BatchWidth * NumRows * NumCols = "
               + std::to_string(np));
         mCmd->sendOk(false);
         delete mCmd;
         throw std::runtime_error("Conflicting values for number of MPI processes");
      }
      else {
         mCmd->sendOk(true);
      }
   }
   else {
      if (!mCmd->waitForOk()) {
         delete mCmd;
         exit(EXIT_FAILURE);
      }
   }
}

PythonBindings::~PythonBindings() {
   delete mCmd;
}

// These methods are called from Python

void PythonBindings::begin() {
   mCmd->begin();
}

double PythonBindings::advance(unsigned int steps) {
   return mCmd->advance(steps); 
}

void PythonBindings::finish() {
   mCmd->finish();
}

py::array_t<float> PythonBindings::getConnectionWeights(const char *connName) {
   std::vector<float> temp;
   int nwp, nyp, nxp, nfp;
   mCmd->getConnectionWeights(connName, &temp, &nwp, &nyp, &nxp, &nfp);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nwp, nyp, nxp, nfp}, false);
   return result;
}

void PythonBindings::setConnectionWeights(const char *connName, py::array_t<float> *data) {
   unsigned int N = data->shape()[0]*data->shape()[1]*data->shape()[2]*data->shape()[3];
   std::vector<float> temp(data->data(), data->data() + N);
   mCmd->setConnectionWeights(connName, &temp);
}

// TODO: This is an ugly return, find a better way to do this
py::tuple PythonBindings::getLayerSparseActivity(const char *layerName) {
   std::vector<std::vector<std::pair<float, int>>> temp;
   int nx, ny, nf;
   py::list result;
   mCmd->getLayerSparseActivity(layerName, &temp, &ny, &nx, &nf);
   for (auto v : temp) {
      result.append(v);
   }
   return py::make_tuple(result, nx, ny, nf);
}


py::array_t<float> PythonBindings::getLayerActivity(const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf, nb;
   mCmd->getLayerActivity(layerName, &temp, &nb, &ny, &nx, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nb, ny, nx, nf}, false);
   return result;
}

py::array_t<float> PythonBindings::getLayerState(const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf, nb;
   mCmd->getLayerState(layerName, &temp, &nb, &ny, &nx, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nb, ny, nx, nf}, false);
   return result;
}

void PythonBindings::setLayerState(const char *layerName, py::array_t<float> *data) {
   unsigned int N = data->shape()[0]*data->shape()[1]*data->shape()[2]*data->shape()[3];
   std::vector<float> temp(data->data(), data->data() + N);
   mCmd->setLayerState(layerName, &temp);
}

bool PythonBindings::isFinished() {
   return mCmd->isFinished();
}

py::array_t<double> PythonBindings::getProbeValues(const char *probeName) {
   std::vector<double> temp;
   mCmd->getProbeValues(probeName, &temp);
   py::array_t<double> result(temp.size(), temp.data());
   return result;
}

bool PythonBindings::isRoot() {
   return mCmd->isRoot();
}

void PythonBindings::waitForCommands() {
   mCmd->waitForCommands();
}


} /* namespace PV */