import time
import traceback
from queue import Empty
from multiprocessing import Process, Queue, Event, Queue, Manager
import sys
sys.path.append('/home/athresher/projects/OpenPV/python')
import OpenPV

from enum import Enum


##############################################################################
# Enum to wrap function callbacks for fetching PetaVision data
##############################################################################

class PVType(Enum):
    LAYER_A    = (OpenPV.PetaVision.get_layer_activity,)
    LAYER_V    = (OpenPV.PetaVision.get_layer_state,)
    CONNECTION = (OpenPV.PetaVision.get_connection_weights,)
    PROBE      = (OpenPV.PetaVision.get_probe_values,)
    def __init__(self, callback):
        self.callback = callback
 

##############################################################################
#  Entry in the list of PetaVision object watch events
##############################################################################

class _WatchEntry:
    def __init__(self, name, pv_type):
        if not isinstance(pv_type, PVType):
            raise TypeError('pv_type must be a PVType enum')
        self.name      = name
        self.callback = pv_type.callback 


##############################################################################
# Entry in the cache of data returned by watch events
##############################################################################

class _CacheEntry:
    def __init__(self, name, data, timestep):
        self.name      = name
        self.data      = data
        self.timestep = timestep


##############################################################################
# On non-root processes, Runner.analyze returns an instance this class instead
# of AnalysisProcess so that chained .watch calls don't throw an error
##############################################################################
class _DummyAnalysisProcess:
    def __init__(self, *args, **kwargs):
        pass
    def watch(self, *args, **kwargs):
        return self


##############################################################################
# Wrapper for analysis processes that run alongside PetaVision
##############################################################################

class AnalysisProcess:
    def __init__(self, callback, interval, queue):
        self._callback = callback
        self._stop     = Event()
        self._start    = Event()
        self._proc     = None
        self._queue    = queue 
        self._watches  = []
        self._interval = interval
        self._counter  = 0
        self.name      = callback.__name__


    def stop(self):
        self._stop.set()
        self._proc.join()



    def start(self):
        self._start.set()

    def launch(self):
        if len(self._watches) == 0:
            print('Warning, analysis process \'%s\' has no watch entries'
                    % (self.name))
        self._proc = Process(
                name   = self.name,
                target = AnalysisProcess._run,
                args   = (self._queue, self._start, self._stop, self._callback))
        self._proc.start()
        print('Launching analysis process \'%s\' with pid %d'
                % (self.name, self._proc.pid))

    def next_update(self):
        return self._counter

    def update(self, steps, pv, timestep):
        self._counter -= steps
        if self._counter <= 0:
            self._counter = self._interval
            entry_list = []
            for w in self._watches:
                data = w.callback(pv, w.name)
                entry_list.append(_CacheEntry(
                    w.name,
                    w.callback(pv, w.name),
                    timestep))
            if len(entry_list) > 0:
                self._queue.put(entry_list)


    def watch(self, obj_name, pv_type):
        if obj_name in self._watches:
            raise ValueError(
                    'analysis process \'%s\' already has a watch entry for \'%s\''
                    % (self.name, obj_name))
        self._watches.append(_WatchEntry(obj_name, pv_type))
        print('Added watch entry for object \'%s\' to process \'%s\'' % (obj_name, self.name))
        return self


    @staticmethod
    def _run(queue, start_event, stop_event, callback):
        # Store the most recent data from each watch entry in a dictionary
        # passed to the callback function
        kwargs = {'simtime' : 0}

        start_event.wait()

        while not stop_event.is_set():
            while queue.empty():
                time.sleep(0.01)

            # This clobbers previous entries, so the callback always gets
            # the most recent state of the entry. Is that what we want?
            while not queue.empty():
                try:
                    entry_list = queue.get(block=False)
                    for e in entry_list:
                        kwargs[e.name] = e.data
                        kwargs['simtime'] = max(kwargs['simtime'], e.timestep)
                except Empty:
                    continue 
            try:
                callback(**kwargs)
            except Exception as e:
                e_name = type(e).__name__
                trace_str = ''.join(traceback.format_tb(e.__traceback__))
                print('*** ' + e_name + ' in ' + callback.__name__ + ' process ***\n' + str(e))
                print(trace_str)

        print('Exiting analysis process \'%s\'' % (str(callback.__name__)))


##############################################################################
# Container object that manages running PetaVision, watching PetaVision objs,
# and running analysis processes
##############################################################################

class Runner:
    def __init__(
            self,
            args,                             # dictionary passed to PetaVision constructor
            params):                         # string containing petavision params
        self._procs    = {}
        self._watchlist = []
        self._time        = 0
        self._pv = OpenPV.PetaVision(args, params)
        self._manager = Manager()

    # Read only access to the current timestep
    def timestep(self):
        return self._time


    # Adds a process that runs the given callback at the given interval
    def analyze(
            self,
            callback,
            interval):
        if self._pv.is_root():
            if callback.__name__ not in self._procs:
                self._procs[callback.__name__] = AnalysisProcess(callback, interval, self._manager.Queue())
                print('Added analysis process \'%s\'' % (callback.__name__))
                return self._procs[callback.__name__]
            else:
                raise ValueError('Cannot add duplicate entry for \'%s\'' % (callback.__name__))
        else:
            return _DummyAnalysisProcess()

    def _update_processes(self, steps):
        for p in self._procs:
            self._procs[p].update(steps, self._pv, self._time)


    # Starts the analysis process and runs the PetaVision network to completion
    def run(self):
        if self._pv.is_root():
            self._pv.begin()

            # Launch each process 
            for p in self._procs:
                self._procs[p].launch()

            try:
                self._update_processes(0)
                for p in self._procs:
                    self._procs[p].start()
                while self._pv.is_finished() == False:
                    min_steps = 1
                    if len(self._procs) > 0:
                        min_steps = min(self._procs[p].next_update() for p in self._procs) 
                    self._time = self._pv.advance(int(min_steps))
                    self._update_processes(min_steps)

            except Exception as e:
                print('*** Error ***\n' + str(type(e)) + '\n' + str(e))
                raise e

            finally:
                for p in self._procs:
                    self._procs[p].stop()
                self._pv.finish()

        else: # not root
            try:
                self._pv.wait_for_commands()
            except Exception as e:
                print('*** Error ***\n' + str(type(e)) + '\n' + str(e))
                raise e
            finally:
                sys.exit()


    def is_root(self):
        return self._pv.is_root()

