import pandas as pd
import matplotlib.pyplot as plt

import sys

if __name__ == '__main__':
    model_name = sys.argv[1]
    df = pd.read_csv("{dir}/iteration_{iter}.csv".format(dir="evaluation_results"
                                                         , iter=model_name), usecols=[1, 2, 3, 4])

    ax = plt.gca()

    # df.plot(kind='line', x='source_azimuth', y='prediction - source_azimuth', ax=ax)
    df.plot(kind='scatter', x='source_azimuth', y='prediction - source_azimuth', color='red')

    # df[['prediction - source_azimuth']].plot(kind='hist', bins=[0, 5, 10], rwidth=0.8)

    plt.show()
