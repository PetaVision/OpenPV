/*
 * pypv.cpp
 *
 *  Created on: Aug 16, 2008
 *      Author: dcoates
 */

// High-level python interface to Petavision
// For now, just provide a single entry-point,
// and no special types or anything fancy like
// that.

#include <Python.h>
#include "../pv_common.h"
#include "../columns/HyPerCol.hpp"
#include "pv_ifc.h"

static PyObject* pypvError;

void simulate_cmdline(int nargs, PyObject* arg, char ***buf)
{
	char *string = PyString_AsString(arg);
	char *ptr, *start;
	int args=0;

	*buf = (char**)malloc(sizeof(char**)*nargs);

	// Won't work if completely empty...
	for (args=0, start=ptr=string; *ptr != 0;  ptr++)
		if (*ptr==' ')
		{
			*ptr=0;
			(*buf)[args++]=start;
			start = ptr+1;
		}

	(*buf)[args++]=start;
}

int pyToFloats(PyObject *list, float **a)
{
	int num = PyTuple_Size(list);
	PyObject* item;
	int i;

	*a = (float*) malloc(num * sizeof(float)); //this can be freed, which MATLAB should do

	for (i = 0; i < num; i++)
	{
		item = PyTuple_GetItem(list, i);
		if (PyFloat_Check(item))
			(*a)[i] = (float) PyFloat_AsDouble(item);
		else if (PyInt_Check(item))
			(*a)[i] = (float) PyInt_AsLong(item);
		else
			return -1;
	}

	return num;
}

int pyInject(PyObject *list, float *buf )
{
	int num = PyList_Size(list);
	PyObject* item;

	if (!PyList_Check(list))
		return NULL;

	int i;
	for (i = 0; i < num; i++)
	{
		item = PyList_GetItem(list, i);
		if (PyFloat_Check(item))
			buf[i] = (float) PyFloat_AsDouble(item);
		else if (PyInt_Check(item))
			buf[i] = (float) PyInt_AsLong(item);
		else
			return -1;
	}

	return num;
}

static PyObject *
pypv_command(PyObject *self, PyObject *args)
{
	// PV_HC Singleton:
	static PV::HyPerCol *hc;
	static char inputFilename[MAX_FILENAME] =
	{ 0 };

	int n = 1;
	int ret, num;
	float carg1[5];
	float *buf;
	void *params;
	float steps, action;
	int value, size;
	PyObject *arg1, *arg2;
	char **cmdline_args;
	// TODO: pass in the global image specs
	PVRect imageRect = {0.0, 0.0, 64.0, 64.0};
	char msg[32];

	// Make all params float, to be like Octave/MATLAB

	num = PyTuple_Size(args);
	if (!PyArg_ParseTuple(args, "fOO", &action, &arg1, &arg2))
		return NULL;

	switch ((int) action)
	{
	case PV_ACTION_INIT:
		value =  (int) PyInt_AsLong(arg1); // nargs
		if (PyString_Check(arg2))
			simulate_cmdline(value, arg2, &cmdline_args);
		else
			return NULL;

		if (hc) delete hc;
		hc = new PV::HyPerCol(&value, &cmdline_args, MAX_LAYERS, MAX_CONNECTIONS, imageRect);
		free(cmdline_args);
		ret = 1;
		break;

	case PV_ACTION_FINALIZE:
		delete hc;
		ret = 1;
		break;

	case PV_ACTION_SETUP:
		ret = hc->init();
		break;

	case PV_ACTION_ADD_LAYER:
		if (!PyArg_ParseTuple(arg1, "ffff", &carg1[0], &carg1[1], &carg1[2],
				&carg1[3]))
			return NULL;
		num = carg1[0];
		sprintf(msg, "layer%d", num);
		ret = PV_ifc_addLayer(hc, msg, carg1[0], carg1[1], carg1[2], carg1[3]);
		break;

	case PV_ACTION_SET_LAYER_PARAMS:
		if (!PyArg_ParseTuple(arg1, "ff", &carg1[0], &carg1[1]))
			return NULL;
		num = pyToFloats(arg2, (float **)&params);
		if (num < 0)
			return NULL;

		// Hack: need a better way to get in strings. TODO
		if (carg1[1] == PV_HANDLER_READFILE)
		{
			(((void**)params)[6]) = inputFilename;
		}
		PV_ifc_setParams(hc, carg1[0], num, (void*) params, carg1[1]);
		free(params);
		ret = 1;
		break;

	case PV_ACTION_ADD_CONNECTION:
		if (!PyArg_ParseTuple(arg1, "fffff", &carg1[0], &carg1[1], &carg1[2],
				&carg1[3], &carg1[4]))
			return NULL;
		num = pyToFloats(arg2, (float **)&params);
		if (num < 0)
			return NULL;
		PV_ifc_connect(hc, carg1[0], carg1[1], carg1[2], carg1[3], num,
				(void*) params, carg1[4]);
		free(params);
		ret = 1;
		break;

	case PV_ACTION_RUN:
		if (PyFloat_Check(arg1))
			steps = (float) PyFloat_AsDouble(arg1);
		else if (PyInt_Check(arg1))
			steps = (float) PyInt_AsLong(arg1);
		else
			return NULL;
		hc->run(steps);
		ret = 1;
		break;

	case PV_ACTION_SET_PARAMS:
		ret = 0;
		return NULL;
		break;

	case PV_ACTION_SET_INPUT_FILENAME:
	{
		char *string;
		if (PyString_Check(arg1))
			string = PyString_AsString(arg1);
		else
			return NULL;

		strcpy(inputFilename, string );
		ret = 1;
		break;
	}

	case PV_ACTION_INJECT:
		// Get 'which buffer' params
		if (!PyArg_ParseTuple(arg1, "fff", &carg1[0], &carg1[1], &carg1[2]))
			return NULL;
		PV_ifc_getBufferPtr(hc, carg1[0],carg1[1],carg1[2], &buf, &size);
		pyInject(arg2, buf);
		break;

	default:
		ret = -1;
		break;
	}

	return Py_BuildValue("i", n);
}

static PyMethodDef pypvMethods[] =
{
{ "command", pypv_command, METH_VARARGS,
		"Execute a Petavision command, such as:\n"
			"PV_ACTION_INIT 1\n"
			"PV_ACTION_ADD_LAYER 2\n"
			"PV_ACTION_SET_LAYER_PARAMS 3\n"
			"PV_ACTION_ADD_CONNECTION 4\n"
			"PV_ACTION_RUN 5\n"
			"PV_ACTION_SET_PARAMS 6\n" "Set input file 7\n" },
{ NULL, NULL, 0, NULL } };

PyMODINIT_FUNC initpypv(void)
{
	PyObject *mod;
	mod = Py_InitModule("pypv", pypvMethods);
	if (mod == NULL)
		return;

	pypvError = PyErr_NewException((char *) "pypv.error", NULL, NULL);
	Py_INCREF(pypvError);
	PyModule_AddObject(mod, "error", pypvError);
}

