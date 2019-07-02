#ifndef PYTHONBINDINGS_HPP_
#define PYTHONBINDINGS_HPP_

#define PYTHON_MODULE_NAME      PythonBindings 
#define PYTHON_MODULE_NAME_STR "PythonBindings"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bindings/Commander.hpp>


namespace py = pybind11;

namespace PV {

/*
   The PythonBindings class provides a Python interface for the Commander 
   class to be used with Python types and structures.
*/

class PythonBindings {
  public:
   PythonBindings(py::dict args, std::string params);
   ~PythonBindings();

  private:
   Commander *mCmd;
   
  public:
   void                begin();
   double              advance(unsigned int steps);
   void                finish();
   py::array_t<float>  getConnectionWeights(const char *connName); 
   void                setConnectionWeights(const char *connName, py::array_t<float> *data); 
   py::array_t<float>  getLayerActivity(const char *layerName); 
   py::array_t<float>  getLayerState(const char *layerName); 
   void                setLayerState(const char *layerName, py::array_t<float> *data);
   bool                isFinished();
   py::array_t<double> getProbeValues(const char *probeName);
   bool                isRoot();
   void                waitForCommands();
};


} /* namespace PV */

PYBIND11_MODULE( PYTHON_MODULE_NAME, m ) {
   m.doc() = "Python bindings for OpenPV";

   py::class_<PV::PythonBindings>(m, "Petavision")
      .def(py::init<py::dict, std::string>())
      .def("begin",                   &PV::PythonBindings::begin)
      .def("advance",                 &PV::PythonBindings::advance)
      .def("finish",                  &PV::PythonBindings::finish)
      .def("getConnectionWeights",    &PV::PythonBindings::getConnectionWeights)
      .def("setConnectionWeights",    &PV::PythonBindings::setConnectionWeights)
      .def("getLayerActivity",        &PV::PythonBindings::getLayerActivity)
      .def("getLayerState",           &PV::PythonBindings::getLayerState)
      .def("setLayerState",           &PV::PythonBindings::setLayerState)
      .def("isFinished",              &PV::PythonBindings::isFinished)
      .def("getProbeValues",          &PV::PythonBindings::getProbeValues)
      .def("isRoot",                  &PV::PythonBindings::isRoot)
      .def("waitForCommands",         &PV::PythonBindings::waitForCommands)
      ;
}


#endif
