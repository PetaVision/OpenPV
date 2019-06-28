#include "PVPython.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

namespace PV {

void errCallback(std::string const err) {
   py::print("[Error] " + err);
}

PythonContext *PyCreateContext(py::dict args, std::string params) {
   PythonContext *pc = new PythonContext();
   for (auto ent : args) {
      args[ent.first] = py::str(ent.second);
   }
   pc->mCmd = new Commander(py::cast<std::map<std::string, std::string>>(args), params, &errCallback);
   return pc;
}

void PyBeginRun(PythonContext *pc) {
   pc->mCmd->beginRun();
}

void PyWaitForCommands(PythonContext *pc) {
   pc->mCmd->waitForCommands();
}

double PyAdvanceRun(PythonContext *pc, unsigned int steps) {
   return pc->mCmd->advanceRun(steps); 
}

void PyFinishRun(PythonContext *pc) {
   pc->mCmd->finishRun();
}

py::array_t<float> PyGetConnectionWeights(PythonContext *pc, const char *connName) {
   std::vector<float> temp;
   int nwp, nyp, nxp, nfp;
   pc->mCmd->getConnectionWeights(connName, &temp, &nwp, &nyp, &nxp, &nfp);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nwp, nyp, nxp, nfp}, false);
   return result;
}

void PySetConnectionWeights(PythonContext *pc, const char *connName, py::array_t<float> *data) {
   unsigned int N = data->shape()[0]*data->shape()[1]*data->shape()[2]*data->shape()[3];
   std::vector<float> temp(data->data(), data->data() + N);
   pc->mCmd->setConnectionWeights(connName, &temp);
}

py::array_t<float> PyGetLayerActivity(PythonContext *pc, const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf, nb;
   pc->mCmd->getLayerActivity(layerName, &temp, &nb, &ny, &nx, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nb, ny, nx, nf}, false);
   return result;
}

py::array_t<float> PyGetLayerState(PythonContext *pc, const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf, nb;
   pc->mCmd->getLayerState(layerName, &temp, &nb, &ny, &nx, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nb, ny, nx, nf}, false);
   return result;
}

void PySetLayerState(PythonContext *pc, const char *layerName, py::array_t<float> *data) {
   unsigned int N = data->shape()[0]*data->shape()[1]*data->shape()[2]*data->shape()[3];
   std::vector<float> temp(data->data(), data->data() + N);
   pc->mCmd->setLayerState(layerName, &temp);
}

bool PyIsFinished(PythonContext *pc) {
   return pc->mCmd->isFinished();
}

py::array_t<double> PyGetProbeValues(PythonContext *pc, const char *probeName) {
   std::vector<double> temp;
   pc->mCmd->getProbeValues(probeName, &temp);
   py::array_t<double> result(temp.size(), temp.data());
   return result;
}

bool PyIsRoot(PythonContext *pc) {
   return pc->mCmd->isRoot();
}

} /* namespace PV */
