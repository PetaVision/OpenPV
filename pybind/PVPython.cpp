#include "PVPython.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

namespace PV {

PythonContext::PythonContext(py::dict args, std::string params) {
   for (auto ent : args) {
      args[ent.first] = py::str(ent.second);
   }
   mCmd = new Commander(py::cast<std::map<std::string, std::string>>(args), params, nullptr);
}

PythonContext::~PythonContext() {
   delete mCmd;
}

void PythonContext::Begin() {
   mCmd->begin();
}

void PythonContext::WaitForCommands() {
   mCmd->waitForCommands();
}

double PythonContext::Advance(unsigned int steps) {
   return mCmd->advance(steps); 
}

void PythonContext::Finish() {
   mCmd->finish();
}

py::array_t<float> PythonContext::GetConnectionWeights(const char *connName) {
   std::vector<float> temp;
   int nwp, nyp, nxp, nfp;
   mCmd->getConnectionWeights(connName, &temp, &nwp, &nyp, &nxp, &nfp);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nwp, nyp, nxp, nfp}, false);
   return result;
}

void PythonContext::SetConnectionWeights(const char *connName, py::array_t<float> *data) {
   unsigned int N = data->shape()[0]*data->shape()[1]*data->shape()[2]*data->shape()[3];
   std::vector<float> temp(data->data(), data->data() + N);
   mCmd->setConnectionWeights(connName, &temp);
}

py::array_t<float> PythonContext::GetLayerActivity(const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf, nb;
   mCmd->getLayerActivity(layerName, &temp, &nb, &ny, &nx, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nb, ny, nx, nf}, false);
   return result;
}

py::array_t<float> PythonContext::GetLayerState(const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf, nb;
   mCmd->getLayerState(layerName, &temp, &nb, &ny, &nx, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nb, ny, nx, nf}, false);
   return result;
}

void PythonContext::SetLayerState(const char *layerName, py::array_t<float> *data) {
   unsigned int N = data->shape()[0]*data->shape()[1]*data->shape()[2]*data->shape()[3];
   std::vector<float> temp(data->data(), data->data() + N);
   mCmd->setLayerState(layerName, &temp);
}

bool PythonContext::IsFinished() {
   return mCmd->isFinished();
}

py::array_t<double> PythonContext::GetProbeValues(const char *probeName) {
   std::vector<double> temp;
   mCmd->getProbeValues(probeName, &temp);
   py::array_t<double> result(temp.size(), temp.data());
   return result;
}

bool PythonContext::IsRoot() {
   return mCmd->isRoot();
}

} /* namespace PV */
