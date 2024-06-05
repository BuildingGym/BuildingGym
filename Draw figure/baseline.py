import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
if __name__ == '__main__':
    ctr_feq = 6
    days = 92
    results = pd.read_csv('Data\\baseline-25.csv')
    ts = pd.date_range("2018-01-01 00:00:00", periods=24*6, freq="10min")
    main = plt.figure()
    daily_energy = []
    for i in range(days):
        day_of_week = results['day_of_week'][144*i]
        if day_of_week<5:
            energy_i = results['cooling_energy'][(i*144):(i+1)*144]
            plt.plot(ts, energy_i, color = (0.1, 0.1, 0.1, 0.3))
            daily_energy.append(energy_i)
    daily_energy = np.array(daily_energy)
    day_mean = daily_energy.mean(axis=0)
    plt.plot(ts, day_mean, color = 'red')
    plt.savefig('Figures\day_mean.png')
    day_mean = pd.DataFrame({'Time': ts, 'Day_mean':day_mean})
    day_mean.to_csv('Data\Day_mean.csv')
    a = 1
