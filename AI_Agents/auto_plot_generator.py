"""
Auto Plot Generator - Integrazione nel Training Loop
Questo modulo può essere importato negli script di training per generare
automaticamente grafici ogni N episodi.
"""

import os
import subprocess
import sys


def auto_generate_plots(log_file, checkpoint_dir, algorithm='DQN', window=20):
    """
    Genera automaticamente grafici dal log di training.

    Args:
        log_file (str): Path al file training_log.txt
        checkpoint_dir (str): Directory dove salvare i grafici
        algorithm (str): 'DQN' o 'PPO'
        window (int): Finestra per smoothing
    """
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Verifica che il log file esista
    if not os.path.exists(log_file):
        print(f"[Auto Plot] Warning: Log file non trovato: {log_file}")
        return

    # Esegui lo script di generazione grafici
    script_path = os.path.join(os.path.dirname(__file__), 'generate_plots.py')

    if not os.path.exists(script_path):
        print(f"[Auto Plot] Warning: Script generate_plots.py non trovato")
        return

    try:
        print(f"\n[Auto Plot] Generazione grafici in corso...")
        result = subprocess.run(
            [
                sys.executable, 
                script_path,
                '--log', log_file,
                '--type', algorithm.lower(),
                '--output', plots_dir,
                '--window', str(window)
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"[Auto Plot] ✓ Grafici generati in: {plots_dir}")
        else:
            print(f"[Auto Plot] ✗ Errore durante generazione:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print(f"[Auto Plot] ✗ Timeout durante generazione grafici")
    except Exception as e:
        print(f"[Auto Plot] ✗ Errore: {e}")
