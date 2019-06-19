#ifndef PVPYTHON_HPP_
#define PVPYTHON_HPP_

#define PYTHON_MODULE_NAME      PVPython
#define PYTHON_MODULE_NAME_STR "PVPython"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <columns/PV_Init.hpp>
#include <columns/HyPerCol.hpp>

namespace py = pybind11;

namespace PV {

struct InteractiveContext {
   InteractiveContext() {}
   int argc;
   char **argv;
   PV_Init *initObj;
   HyPerCol *hc;
};

InteractiveContext *newContext(py::dict args, std::string params);
void                freeContext(InteractiveContext *ic);
void                beginRun(InteractiveContext *ic);
double              advanceRun(InteractiveContext *ic, unsigned int steps);
void                finishRun(InteractiveContext *ic);
py::array_t<float>  getLayerActivity(InteractiveContext *ic, const char *layerName); 

}

PYBIND11_MODULE( PYTHON_MODULE_NAME, m ) {
   m.doc() = "Python bindings for OpenPV";

   py::class_<PV::InteractiveContext>(m, "PVContext")
      .def(py::init<>());

   m.def("beginRun",    &PV::beginRun);
   m.def("advanceRun",  &PV::advanceRun);
   m.def("finishRun",   &PV::finishRun);
   m.def("newContext",  &PV::newContext, py::return_value_policy::reference);
   m.def("freeContext", &PV::freeContext);
   m.def("getLayerActivity", &PV::getLayerActivity);
}


#endif
