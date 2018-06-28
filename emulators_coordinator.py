import numpy as np
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double, c_int32, c_bool
from emulator_runner import EmulatorRunner

class SimulatorsCoordinator(object):
    """
    It controls worker processes which in turn control emulators.
    - Instantiated by master training process.
    - Instantiates and makes available queue based control of multiple worker
    processes.
    - Also creates and returns values across shared memory variables filled in
    by emulators.
    - Each worker process instantiates and steps through multiple emulators.
    - Each emulator is associated with requisite shared memory variables, which
    get filled in as worker process steps through them.
    """
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.int32: c_int32, np.uint8: c_uint, np.bool_: c_bool}

    def __init__(self, environment_creator, n_emulators_per_emulator_runner, emulator_runners, variables):
        self.variables = {k: self._get_shared(var) for k, var in variables.items()}
        self.emulator_runners = emulator_runners
        self.queues = [Queue() for _ in range(self.emulator_runners)]
        self.barrier = Queue()

        # Create pointers to slices of the the array-buffer
        sim_vars = [{k: var[i * n_emulators_per_emulator_runner: (i + 1) * n_emulators_per_emulator_runner] for k, var in self.variables.items()} for i in range(emulator_runners)]

        self.simulators = [EmulatorRunner(i, environment_creator, n_emulators_per_emulator_runner, sim_vars[i], self.queues[i], self.barrier) for i in range(emulator_runners)]
        print(len(self.simulators))

    def _get_shared(self, array):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :return: the RawArray backed numpy array
        """

        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def start(self):
        for w in self.simulators:
            w.start()

    def stop(self):
        for queue in self.queues:
            queue.put(None)

    def get_shared_variables(self):
        return self.variables

    def update_environments(self):
        for queue in self.queues:
            queue.put(True)

    def wait_updated(self):
        for wd in range(self.emulator_runners):
            self.barrier.get()
