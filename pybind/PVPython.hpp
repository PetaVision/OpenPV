#ifndef PVPYTHON_HPP_
#define PVPYTHON_HPP_

#define PYTHON_MODULE_NAME      PVPython 
#define PYTHON_MODULE_NAME_STR "PVPython"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bindings/Commander.hpp>


namespace py = pybind11;

namespace PV {

class PythonContext {
  public:
   PythonContext(py::dict args, std::string params);
   ~PythonContext();

  private:
   Commander *mCmd;
   
  public:
   void                Begin();
   double              Advance(unsigned int steps);
   void                Finish();
   py::array_t<float>  GetConnectionWeights(const char *connName); 
   void                SetConnectionWeights(const char *connName, py::array_t<float> *data); 
   py::array_t<float>  GetLayerActivity(const char *layerName); 
   py::array_t<float>  GetLayerState(const char *layerName); 
   void                SetLayerState(const char *layerName, py::array_t<float> *data);
   bool                IsFinished();
   py::array_t<double> GetProbeValues(const char *probeName);
   bool                IsRoot();
   void                WaitForCommands();
};


} /* namespace PV */

PYBIND11_MODULE( PYTHON_MODULE_NAME, m ) {
   m.doc() = "Python bindings for OpenPV";

   py::class_<PV::PythonContext>(m, "PVContext")
      .def(py::init<py::dict, std::string>())
      .def("begin",                   &PV::PythonContext::Begin)
      .def("advance",                 &PV::PythonContext::Advance)
      .def("finish",                  &PV::PythonContext::Finish)
      .def("getConnectionWeights",    &PV::PythonContext::GetConnectionWeights)
      .def("setConnectionWeights",    &PV::PythonContext::SetConnectionWeights)
      .def("getLayerActivity",        &PV::PythonContext::GetLayerActivity)
      .def("getLayerState",           &PV::PythonContext::GetLayerState)
      .def("setLayerState",           &PV::PythonContext::SetLayerState)
      .def("isFinished",              &PV::PythonContext::IsFinished)
      .def("getProbeValues",          &PV::PythonContext::GetProbeValues)
      .def("isRoot",                  &PV::PythonContext::IsRoot)
      .def("waitForCommands",         &PV::PythonContext::WaitForCommands)
      ;
}


#endif
