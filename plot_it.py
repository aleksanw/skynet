import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_log_dir(path) -> (pd.DataFrame, int):
    logs = pathlib.Path(path)
    logs = [*map(pd.read_csv, logs.iterdir())]
    log_count = len(logs)
    logs = pd.concat(logs).groupby('env_step_count')
    return logs, log_count


def main():
    logs_base, log_count_base = read_log_dir('progress_logs_baseline')
    logs_shift, log_count_shift = read_log_dir('progress_logs_shift_p5')

    sns.lineplot(data=pd.DataFrame({
        'Baseline': logs_base.mean()['avg_reward'],
        'Shifted': logs_shift.mean()['avg_reward'],
    }))

    plt.show()


if __name__ == '__main__':
    main()