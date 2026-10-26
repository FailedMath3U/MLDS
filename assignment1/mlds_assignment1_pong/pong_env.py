"""CPU Atari workers; this module deliberately never imports torch."""
import multiprocessing as mp
import traceback
import numpy as np

ACTIONS = (2, 3)


def preprocess(image):
    """Assignment crop/downsample/background removal, without mutating input."""
    frame = image[35:195:2, ::2, 0].copy()
    frame[(frame == 144) | (frame == 109)] = 0
    return (frame != 0).astype(np.int8)


def make_env():
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make('ALE/Pong-v5')
    names = env.unwrapped.get_action_meanings()
    if names[2:4] != ['RIGHT', 'LEFT']:
        raise RuntimeError(f'Unexpected action map: {names}')
    return env


def metadata(env):
    return {'id': env.spec.id, 'kwargs': env.spec.kwargs,
            'max_episode_steps': env.spec.max_episode_steps,
            'action_meanings': env.unwrapped.get_action_meanings()}


def worker(connection):
    env = None
    try:
        env = make_env()
        previous = None
        connection.send(('ok', metadata(env)))
        while True:
            command, value = connection.recv()
            if command == 'close':
                break
            if command == 'reset':
                raw, _ = env.reset(seed=int(value))
                previous = preprocess(raw)
                connection.send(('ok', np.zeros(6400, dtype=np.int8)))
            elif command == 'step':
                if int(value) not in ACTIONS:
                    raise ValueError('Only actions 2 and 3 are permitted, including startup.')
                raw, reward, terminated, truncated, _ = env.step(int(value))
                current = preprocess(raw)
                difference = (current - previous).ravel()
                previous = current
                connection.send(('ok', (difference, float(reward), bool(terminated), bool(truncated))))
            else:
                raise ValueError(command)
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except BaseException:
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class Pool:
    def __init__(self, count):
        self.connections, self.processes = [], []
        context = mp.get_context('spawn')
        try:
            for _ in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(child,), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.metadata = [self.receive(i) for i in range(count)]
        except BaseException:
            self.close()
            raise

    def receive(self, index, timeout=120):
        connection = self.connections[index]
        if not connection.poll(timeout):
            raise TimeoutError(f'Atari worker {index} did not respond within {timeout}s')
        status, result = connection.recv()
        if status != 'ok':
            raise RuntimeError(result)
        return result

    def reset(self, seeds):
        for i, seed in enumerate(seeds):
            self.connections[i].send(('reset', int(seed)))
        return [self.receive(i) for i in range(len(seeds))]

    def step(self, actions):
        # actions is a dictionary: worker index -> permitted ALE action.
        for i, action in actions.items():
            self.connections[i].send(('step', int(action)))
        return {i: self.receive(i) for i in actions}

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(('close', None))
                except (BrokenPipeError, OSError):
                    pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        for connection in self.connections:
            connection.close()
