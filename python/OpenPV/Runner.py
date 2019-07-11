import time
import threading
import queue
import sys
sys.path.append('/home/athresher/projects/OpenPV/python')
import OpenPV

from enum import Enum

class PVType(Enum):
   LAYER_A    = (OpenPV.PetaVision.get_layer_activity,)
   LAYER_V    = (OpenPV.PetaVision.get_layer_state,)
   CONNECTION = (OpenPV.PetaVision.get_connection_weights,)
   PROBE      = (OpenPV.PetaVision.get_probe_values,)
   def __init__(self, callback):
      self.callback = callback

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

class _CacheEntry:
   def __init__(self, name, data, history):
      self.name    = name
      self.data    = data
      self.history = history

class Runner:
   def __init__(
         self,
         args,                      # dictionary passed to PetaVision constructor
         params,                    # string containing petavision params
         analysis_callback,         # callback function called at end of analysis loop
         analysis_sleep_interval=1, # Seconds to sleep inbetween each analysis loop
         print_memory=False):       # Enables messages that estimate memory usage of each cache entry
      self._stop_event        = threading.Event()
      self._analysis_thread   = threading.Thread(
            target=Runner._analysis_thread,
            args=(self, self._stop_event, ))
      self._analysis_callback       = analysis_callback
      self._analysis_sleep_interval = analysis_sleep_interval
      self._watchlist = []
      self._cache = {}
      self._queue = queue.Queue()
      self._time  = 0
      self._print_memory = print_memory
      self._pv = OpenPV.PetaVision(args, params)


   # Advances PetaVision the number of timesteps until the next watch entry update
   def _advance(self):
      min_steps = min(w.counter for w in self._watchlist) 
      self._time = self._pv.advance(min_steps)
      for w in self._watchlist:
         w.counter -= min_steps
         if w.counter <= 0:
            w.counter = w.interval
            e = _CacheEntry(w.name, w.callback(self._pv, w.name), w.history)
            self._queue.put(e)


   # Moves data from the watch queue into the cache for thread safe access
   def _queue_to_cache(self):
      while self._queue.qsize() > 0:
         try:
            entry = self._queue.get(block=False)
            self._cache[entry.name].append(entry.data)
            while len(self._cache[entry.name]) > entry.history:
               self._cache[entry.name].pop(0)
         except queue.Empty:   
            break

   # Analysis loop thread entry point
   def _analysis_thread(self, stop_event):
      print('Beginning analysis thread with sleep interval of %f seconds'
            % (self._analysis_sleep_interval))
      while not stop_event.is_set():
         self._queue_to_cache()
         self._analysis_callback(self)
         time.sleep(self._analysis_sleep_interval)
      print('Exiting analysis thread')

   # Read only access to the current timestep
   def timestep(self):
      return self._time

   # Adds a watch entry for the named PetaVision network object
   def watch(
         self,
         name,                # Name of object in PetaVision params
         timestep_interval,   # Get data from this object every N timesteps
         pv_type,             # Type of object in PetaVision params
         history=1):          # Keep results of N previous watch queries
      if self._pv.is_root():
         if name in self._cache:
            raise ValueError('Object ' + name + ' already has an entry.')
         self._watchlist.append(_WatchEntry(name, timestep_interval, pv_type, history))
         self._cache[name] = []

   # Starts the analysis thread and runs the PetaVision network to completion
   def run(self):
      if self._pv.is_root():
         self._pv.begin()
         # Calculate size of each cache entry and fill each cache entry
         # with the initial state
         if self._print_memory:
            print("Cache memory usage:")
         for w in self._watchlist:
            d = w.callback(self._pv, w.name)
            e = _CacheEntry(w.name, d, w.history)
            if self._print_memory:
               b = d.nbytes * w.history
               suf = " bytes"
               if b > 1024 * 10:
                  b /= 1024
                  suf = " KB"
               elif b > 1024 * 1024 * 10:
                  b /= 1024 * 1024
                  suf = " MB"
               print("\t" + (str(b) + suf).ljust(24) + w.name)
            self._cache[w.name].append(d)

         self._analysis_thread.start()
         while self._pv.is_finished() == False:
            self._advance()
         self._stop_event.set()
         self._pv.finish()
      else:
         self._pv.wait_for_commands()
         sys.exit()

   def get(self, name):
      return self._cache[name]
