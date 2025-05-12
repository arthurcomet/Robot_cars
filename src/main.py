from client.robocar_client import RobocarClient
from input.input_manager import InputManager
from data.data_collector import DataCollector
from ai.robocar_ai import RobocarAI
import time


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
                    obs = client.get_observations()
                    data_collector.collect(obs, input_manager.current_speed, input_manager.current_steering)
                    time.sleep(0.05)
            except KeyboardInterrupt:
                input_manager.stop()
                data_collector.save()
                data_collector.analyze()

        elif choice == '3':
            if data_collector.df is None:
                print("Aucune donnée disponible. Veuillez d'abord collecter des données.")
                continue
            X = data_collector.df.iloc[:, :-2].values
            y = data_collector.df.iloc[:, -2:].values
            ai.build_model(X.shape[1])
            ai.train(X, y)
            loss, mae = ai.evaluate(X, y)
            print(f"\nPerformance du modèle : Loss = {loss:.4f}, MAE = {mae:.4f}")

        elif choice == '4':
            if ai.model is None:
                print("L'IA n'a pas encore été entraînée. Veuillez d'abord entraîner le modèle.")
                continue
            print("\nDébut de la conduite autonome...")
            try:
                while True:
                    obs = client.get_observations()
                    if obs is not None:
                        X = obs[0].flatten()[:10].reshape(1, -1)
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