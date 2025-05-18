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

    menu_actions = {
        '1': lambda: run_manual_mode(input_manager),
        '2': lambda: run_data_collection(client, input_manager, data_collector),
        '3': lambda: train_ai(data_collector, ai),
        '4': lambda: run_autonomous_mode(client, ai),
        '5': exit_program
    }

    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1. Mode de conduite manuelle")
        print("2. Collecte de données")
        print("3. Entraînement de l'IA")
        print("4. Mode de conduite autonome")
        print("5. Quitter")

        choice = input("Choisissez une option (1-5): ").strip()

        action = menu_actions.get(choice)
        if action:
            action()
        else:
            print("Option invalide. Veuillez choisir une option entre 1 et 5.")

    client.close()


def run_manual_mode(input_manager):
    input_manager.start()
    print("Mode manuel activé. Appuyez sur Ctrl+C pour quitter.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        input_manager.stop()
        print("Mode manuel terminé.")

def run_data_collection(client, input_manager, data_collector):
    print("\nDébut de la collecte de données... (Ctrl+C pour arrêter)")
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
        print("Collecte de données terminée et sauvegardée.")

def train_ai(data_collector, ai):
    if data_collector.df is None:
        print("Aucune donnée disponible. Veuillez d'abord collecter des données.")
        return
    X = data_collector.df.iloc[:, :-2].values
    y = data_collector.df.iloc[:, -2:].values
    ai.build_model(X.shape[1])
    ai.train(X, y)
    loss, mae = ai.evaluate(X, y)
    print(f"\nModèle entraîné : Loss = {loss:.4f}, MAE = {mae:.4f}")

def run_autonomous_mode(client, ai):
    if ai.model is None:
        print("L'IA n'a pas encore été entraînée.")
        return
    print("\nDébut de la conduite autonome... (Ctrl+C pour arrêter)")
    try:
        while True:
            obs = client.get_observations()
            if obs is not None:
                X = obs[0].flatten()[:10].reshape(1, -1)
                actions = ai.predict(X)
                client.send_actions(actions)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Conduite autonome arrêtée.")

def exit_program():
    print("Fermeture du programme...")
    sys.exit(0)



if __name__ == "__main__":
    main()