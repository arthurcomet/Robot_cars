##
## EPITECH PROJECT, 2024
## robotCars
## File description:
## main.py
##

import os
import json
import keyboard
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import threading
import sys
import curses
import select
import termios
import tty
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class RobocarClient:
    def __init__(self, port=5004):
        self.port = port
        self.env = None
        self.engine_config_channel = EngineConfigurationChannel()
        self.config_path = "config.json"
        self.agents = []
        
    def create_config(self):
        config = {
            "agents": [
                {
                    "fov": 180,
                    "nbRay": 10
                },
                {
                    "fov": 48,
                    "nbRay": 36
                }
            ]
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def connect(self):
        try:
            self.create_config()
            self.engine_config_channel.set_configuration_parameters(
                width=1280,
                height=720,
                quality_level=5,
                time_scale=1.0,
                target_frame_rate=60,
                capture_frame_rate=60
            )
            self.env = UnityEnvironment(
                file_name="./RacingSimulator.x86_64",
                side_channels=[self.engine_config_channel],
                base_port=self.port,
                no_graphics=False,
                worker_id=0,
                additional_args=[
                    "--config-path", self.config_path,
                    "--screen-fullscreen", "0",
                    "--screen-width", "1280",
                    "--screen-height", "720"
                ]
            )
            self.env.reset()
            behavior_names = list(self.env.behavior_specs.keys())
            if behavior_names:
                self.primary_behavior = behavior_names[0]
                behavior_spec = self.env.behavior_specs[self.primary_behavior]
                return True
            return False
        except Exception as e:
            print(f"Erreur de connexion: {e}")
            return False
    
    def get_observations(self):
        if self.env:
            decision_steps, terminal_steps = self.env.get_steps(self.primary_behavior)
            if len(decision_steps) > 0:
                return decision_steps[0].obs
        return None
    
    def send_actions(self, actions):
        if self.env:
            action_tuple = ActionTuple(continuous=actions)
            self.env.set_actions(self.primary_behavior, action_tuple)
            self.env.step()
    
    def close(self):
        if self.env:
            self.env.close()


class InputManager:
    def __init__(self, client):
        self.client = client
        self.speed_increment = 0.1
        self.steering_increment = 0.2
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.running = False
        self.thread = None
        self.old_settings = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._input_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
    def _get_key(self):
        try:
            return sys.stdin.read(1)
        except:
            return None
            
    def _input_loop(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        while self.running:
            try:
                key = self._get_key()
                if key:
                    if key == 'q':
                        self.running = False
                        break
                    elif key == '\x1b':
                        next1 = sys.stdin.read(1)
                        next2 = sys.stdin.read(1)
                        if next1 == '[':
                            if next2 == 'A':
                                self.current_speed = min(1.0, self.current_speed + self.speed_increment)
                            elif next2 == 'B':
                                self.current_speed = max(-1.0, self.current_speed - self.speed_increment)
                            elif next2 == 'C':
                                self.current_steering = min(1.0, self.current_steering + self.steering_increment)
                            elif next2 == 'D':
                                self.current_steering = max(-1.0, self.current_steering - self.steering_increment)
                else:
                    if abs(self.current_speed) < 0.05:
                        self.current_speed = 0.0
                    else:
                        self.current_speed *= 0.95
                    if abs(self.current_steering) < 0.05:
                        self.current_steering = 0.0
                    else:
                        self.current_steering *= 0.85
                actions = np.array([[self.current_speed, self.current_steering]], dtype=np.float32)
                self.client.send_actions(actions)
                time.sleep(0.05)
            except Exception as e:
                print(f"Erreur lors de la lecture des touches: {e}")
                self.running = False
                break


class DataCollector:
    def __init__(self):
        self.data = []
        self.columns = ['ray_1', 'ray_2', 'ray_3', 'ray_4', 'ray_5', 'ray_6', 'ray_7', 'ray_8', 'ray_9', 'ray_10',
                       'speed', 'steering']
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
                plt.subplot(3, 4, i+1)
                plt.hist(self.df[col], bins=50)
                plt.title(col)
            plt.tight_layout()
            plt.savefig('data_distribution.png')
            plt.close()


class RobocarAI:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self, input_shape):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='tanh')
        ])
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        X_scaled = self.scaler.transform(X)
        return self.model.evaluate(X_scaled, y, verbose=0)


def main():
    client = RobocarClient(port=5004)
    if not client.connect():
        print("Échec de la connexion au simulateur. Fin du programme.")
        return
    input_manager = InputManager(client)
    data_collector = DataCollector()
    ai = RobocarAI()
    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1. Mode de conduite manuelle")
        print("2. Collecte de données")
        print("3. Entraînement de l'IA")
        print("4. Mode de conduite autonome")
        print("5. Quitter")
        choice = input("Choisissez une option (1-5): ")
        if choice == '1':
            input_manager.start()
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                input_manager.stop()
        elif choice == '2':
            print("\nDébut de la collecte de données...")
            print("Appuyez sur 'Q' pour arrêter la collecte")
            input_manager.start()
            try:
                while True:
                    observations = client.get_observations()
                    data_collector.collect(observations, input_manager.current_speed, input_manager.current_steering)
                    time.sleep(0.05)
            except KeyboardInterrupt:
                input_manager.stop()
                data_collector.save()
                data_collector.analyze()
        elif choice == '3':
            if data_collector.df is None:
                print("Aucune donnée disponible. Veuillez d'abord collecter des données.")
                continue
            print("\nDébut de l'entraînement...")
            X = data_collector.df.iloc[:, :-2].values
            y = data_collector.df.iloc[:, -2:].values
            ai.build_model(X.shape[1])
            history = ai.train(X, y)
            loss, mae = ai.evaluate(X, y)
            print(f"\nPerformance du modèle:")
            print(f"Loss: {loss:.4f}")
            print(f"MAE: {mae:.4f}")
        elif choice == '4':
            if ai.model is None:
                print("L'IA n'a pas encore été entraînée. Veuillez d'abord entraîner le modèle.")
                continue
            print("\nDébut de la conduite autonome...")
            print("Appuyez sur 'Q' pour arrêter")
            try:
                while True:
                    observations = client.get_observations()
                    if observations is not None:
                        X = observations[0].flatten()[:10].reshape(1, -1)
                        actions = ai.predict(X)
                        client.send_actions(actions)
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("Arrêt de la conduite autonome...")
        elif choice == '5':
            print("Fermeture du programme...")
            break
        else:
            print("Option invalide. Veuillez choisir une option entre 1 et 5.")
    client.close()


if __name__ == "__main__":
    main()