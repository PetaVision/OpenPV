#include "PVPython.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

namespace PV {

PythonContext *PyCreateContext(py::dict args, std::string params) {
   PythonContext *pc = new PythonContext();
   for (auto ent : args) {
      args[ent.first] = py::str(ent.second);
   }
   pc->mIC = new InteractiveContext(py::cast<std::map<std::string, std::string>>(args), params);
   return pc;
}

void PyBeginRun(PythonContext *pc) {
   pc->mIC->beginRun();
}

double PyAdvanceRun(PythonContext *pc, unsigned int steps) {
   return pc->mIC->advanceRun(steps); 
}

void PyFinishRun(PythonContext *pc) {
   pc->mIC->finishRun();
}

py::array_t<float> PyGetLayerActivity(PythonContext *pc, const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf;
   pc->mIC->getLayerActivity(layerName, &temp, &nx, &ny, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nx, ny, nf}, false);
   return result;
}

py::array_t<float> PyGetLayerState(PythonContext *pc, const char *layerName) {
   std::vector<float> temp;
   int nx, ny, nf;
   pc->mIC->getLayerState(layerName, &temp, &nx, &ny, &nf);
   py::array_t<float> result(temp.size(), temp.data());
   result.resize({nx, ny, nf}, false);
   return result;
}

void PySetLayerState(PythonContext *pc, const char *layerName, py::array_t<float> *data) {
   unsigned int N = data->shape()[0]*data->shape()[1]*data->shape()[2];
   std::vector<float> temp(data->data(), data->data() + N);
   pc->mIC->setLayerState(layerName, &temp);
}

bool PyIsFinished(PythonContext *pc) {
   return pc->mIC->isFinished();
}

py::array_t<double> PyGetEnergy(PythonContext *pc, const char *probeName) {
   std::vector<double> temp;
   pc->mIC->getEnergy(probeName, &temp);
   py::array_t<double> result(temp.size(), temp.data());
   return result;
}

} /* namespace PV */
