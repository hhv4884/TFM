import os
from core.graph_analysis import create_separated_panels


def process_all_models(root_path, epochs=60):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) != "graph_analysis":
            continue
        graph_path = dirpath
        model_path = os.path.dirname(graph_path)
        logs_path = os.path.join(model_path, "logs")
        if not os.path.exists(logs_path):
            print(f"Logs no encontrados: {model_path}")
            continue
        print("\n" + "=" * 80)
        print(f"MODEL: {model_path}")
        print("=" * 80)
        create_separated_panels(
            model_path=model_path,
            graph_path=graph_path,
            logs_path=logs_path,
            epochs=epochs,
        )


if __name__ == "__main__":
    process_all_models(root_path="trained_models", epochs=60)
