import time
import multiprocessing 
import queue
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
   def __init__(self, name, timestep_interval, pv_type, history=1):
      if history < 1:
         print('WatchEntry.history cannot be < 1. Setting to 1.')
         history = 1

      if not isinstance(pv_type, PVType):
         raise TypeError('pv_type must be a PVType enum')

      self.name     = name
      self.interval = timestep_interval
      self.counter  = timestep_interval
      self.history  = history 
      self.callback = pv_type.callback 


##############################################################################
# Entry in the cache of data returned by watch events
##############################################################################

class _CacheEntry:
   def __init__(self, name, data, history, timestep):
      self.name     = name
      self.data     = data
      self.history  = history
      self.timestep = timestep


##############################################################################
# Wrapper for analysis processes that run alongside PetaVision
##############################################################################

class _AnalysisProcess:
   def __init__(self, callback, seconds, runner):
      self._callback   = callback
      self._seconds    = seconds
      self._stop_event = multiprocessing.Event()
      self._proc     = multiprocessing.Process(
            target=_AnalysisProcess._run,
            args=(self, runner),
            name=str(callback.__name__))

   def stop(self):
      self._stop_event.set()
      self._proc.join()

   def start(self):
      self._proc.start()

   def _run(self, runner):
      print('Beginning analysis process \'%s\' with interval of %f sec'
            % (self._proc.name, self._seconds))
      while not self._stop_event.wait(self._seconds):
         runner.get_cache().update()
         self._callback(runner.timestep(), runner.get_cache())
      print('Exiting analysis process \'%s\'' % (str(self._proc.name)))



##############################################################################
# Cache wrapper
##############################################################################

class _DataCache:
   def __init__(self):
      self._cache = {}
      self._queue = queue.Queue()

   def init_entry(self, name):
      self._cache[name] = []

   def put(self, entry):
      self._queue.put(entry)

   def update(self):
      while self._queue.qsize() > 0:
         try:
            entry = self._queue.get(block=False)
            self._cache[entry.name].append(entry.data)
            while len(self._cache[entry.name]) > entry.history:
               self._cache[entry.name].pop(0)
         except queue.Empty: 
            break
   
   def get(self, name):
      return self._cache[name]


##############################################################################
# Container object that manages running PetaVision, watching PetaVision objs,
# and running analysis processes
##############################################################################

class Runner:
   def __init__(
         self,
         args,                      # dictionary passed to PetaVision constructor
         params,                    # string containing petavision params
         print_memory=False):       # Enables messages that estimate memory usage of each cache entry
      self._procs   = []
      self._watchlist = []
      self._cache     = _DataCache()
      self._time      = 0
      self._print_memory = print_memory
      self._pv = OpenPV.PetaVision(args, params)


   # Advances PetaVision the number of timesteps until the next watch entry update
   def _advance(self):
      min_steps = 1
      if len(self._watchlist) > 0:
         min_steps = min(w.counter for w in self._watchlist) 
      self._time = self._pv.advance(min_steps)
      for w in self._watchlist:
         w.counter -= min_steps
         if w.counter <= 0:
            w.counter = w.interval
            self._cache.put(
                  _CacheEntry(
                     w.name,
                     w.callback(self._pv, w.name),
                     w.history,
                     self._time))


   def _print_memory_usage(self, name, size):
      suf = " bytes"
      if size > 1024 * 10:
         size /= 1024
         suf = " KB"
      elif size > 1024 * 1024 * 10:
         size /= 1024 * 1024
         suf = " MB"
      print("\t" + (str(size) + suf).ljust(24) + name)


   # Read only access to the current timestep
   def timestep(self):
      return self._time

   # Adds a process that runs the given callback at the given interval
   def analyze(
         self,
         callback,
         seconds):
      if self._pv.is_root():
         self._procs.append(_AnalysisProcess(callback, seconds, self))
         print('Added analysis process \'%s\'' % (str(callback.__name__)))

   # Adds a watch entry for the named PetaVision network object
   def watch(
         self,
         name,                # Name of object in PetaVision params
         timestep_interval,   # Get data from this object every N timesteps
         pv_type,             # Type of object in PetaVision params
         history=1):          # Keep results of N previous watch queries
      if self._pv.is_root():
         # TODO: check for duplicate entry
         self._watchlist.append(_WatchEntry(name, timestep_interval, pv_type, history))
         self._cache.init_entry(name)
         print('Added watch entry for object \'%s\'' % (name))

   # Starts the analysis process and runs the PetaVision network to completion
   def run(self):
      if self._pv.is_root():
         self._pv.begin()

         # Calculate size of each cache entry and fill each cache entry
         # with the initial state
         if self._print_memory:
            print("Cache memory usage:")
         for w in self._watchlist:
            d = w.callback(self._pv, w.name)
            e = _CacheEntry(w.name, d, w.history, 0)
            if self._print_memory:
               self._print_memory_usage(w.name, d.nbytes * w.history)
            self._cache.put(e)
         self._cache.update()

         # Start each process 
         for t in self._procs:
            t.start()

         while self._pv.is_finished() == False:
            self._advance()

         # Stop each process 
         for t in self._procs:
            t.stop()

         self._pv.finish()

      else:
         try:
            self._pv.wait_for_commands()
         finally:
            sys.exit()

   def is_root(self):
      return self._pv.is_root()

   def get_cache(self):
      return self._cache
