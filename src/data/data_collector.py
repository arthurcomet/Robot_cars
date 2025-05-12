import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataCollector:
    def __init__(self):
        self.data = []
        self.columns = [f'ray_{i+1}' for i in range(10)] + ['speed', 'steering']
        self.df = None

    def collect(self, observations, speed, steering):
        if observations is not None and len(observations) > 0:
            ray_data = observations[0].flatten()[:10]
            self.data.append(np.concatenate([ray_data, [speed, steering]]))

    def save(self, filename='driving_data.csv'):
        self.df = pd.DataFrame(self.data, columns=self.columns)
        self.df.to_csv(filename, index=False)
        print(f"Données sauvegardées dans {filename}")

    def analyze(self):
        if self.df is not None:
            print(f"Nombre total d'échantillons: {len(self.df)}")
            print(self.df.describe())
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(self.columns):
                plt.subplot(3, 4, i + 1)
                plt.hist(self.df[col], bins=50)
                plt.title(col)
            plt.tight_layout()
            plt.savefig('data_distribution.png')
            plt.close()
