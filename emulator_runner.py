from multiprocessing import Process

class EmulatorRunner(Process):
    def __init__(self, id, env_creator, n_emulators_per_emulator_runner, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.variables = variables
        self.queue = queue
        self.barrier = barrier
        self.initialization_complete = False
        self.emulators = [env_creator.create_environment(i)
                                 for i in range(n_emulators_per_emulator_runner)]
        print("Emulators:", len(self.emulators))

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()


    def _run(self):
        count = 0
        _ = self.queue.get()
        for i, emu in enumerate(self.emulators):
            self.variables["s"][i] = emu.reset()
        self.barrier.put(True)
            
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break

            for i, (emulator, action) in enumerate(zip(self.emulators, self.variables["a"])):
                new_s, reward, episode_over, info = emulator.step(action)
                if episode_over:
                    self.variables["s"][i] = emulator.reset()
                else:
                    self.variables["s"][i] = new_s
                self.variables["r"][i] = reward
                self.variables["done"][i] = episode_over

            count += 1
            self.barrier.put(True)



