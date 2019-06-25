#ifndef PVPYTHON_HPP_
#define PVPYTHON_HPP_

#define PYTHON_MODULE_NAME      PVPython
#define PYTHON_MODULE_NAME_STR "PVPython"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bindings/Commander.hpp>


namespace py = pybind11;

namespace PV {

struct PythonContext {
   PythonContext() {
      mCmd = nullptr;
   }
   ~PythonContext() {
      if (mCmd != nullptr) {
         delete(mCmd);
      }
   }
   Commander *mCmd;
};

PythonContext      *PyCreateContext(py::dict args, std::string params);
void                PyBeginRun(PythonContext *pc);
double              PyAdvanceRun(PythonContext *pc, unsigned int steps);
void                PyFinishRun(PythonContext *pc);
py::array_t<float>  PyGetLayerActivity(PythonContext *pc, const char *layerName); 
py::array_t<float>  PyGetLayerState(PythonContext *pc, const char *layerName); 
//void                PySetLayerState(PythonContext *pc, const char *layerName, py::array_t<float> *data);
bool                PyIsFinished(PythonContext *pc);
py::array_t<double> PyGetEnergy(PythonContext *pc, const char *probeName);
bool                PyIsRoot(PythonContext *pc);
void                PyWaitForCommands(PythonContext *pc);

} /* namespace PV */

PYBIND11_MODULE( PYTHON_MODULE_NAME, m ) {
   m.doc() = "Python bindings for OpenPV";

   py::class_<PV::PythonContext>(m, "PVContext")
      .def(py::init<>());

   m.def("createContext",    &PV::PyCreateContext);
   m.def("beginRun",         &PV::PyBeginRun);
   m.def("advanceRun",       &PV::PyAdvanceRun);
   m.def("finishRun",        &PV::PyFinishRun);
   m.def("getLayerActivity", &PV::PyGetLayerActivity);
   m.def("getLayerState",    &PV::PyGetLayerState);
//   m.def("setLayerState",    &PV::PySetLayerState);
   m.def("isFinished",       &PV::PyIsFinished);
   m.def("getEnergy",        &PV::PyGetEnergy);
   m.def("isRoot",           &PV::PyIsRoot);
   m.def("waitForCommands",  &PV::PyWaitForCommands);
}


#endif
