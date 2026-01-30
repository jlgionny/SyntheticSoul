import os
import sys
import time
import subprocess
import numpy as np
from datetime import datetime
import argparse

# Setup dei path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agents.dqn_agent import DQNAgent
from src.env.hollow_knight_env import HollowKnightEnv


# ============ AUTO PLOT GENERATOR ============
def auto_generate_plots(
    log_file, checkpoint_dir, algorithm="DQN", window=20, current_episode=0
):
    """Genera grafici automaticamente senza bloccare il training."""
    episode_folder = f"episode_{current_episode}"
    plots_dir = os.path.join(checkpoint_dir, "plots", episode_folder)
    os.makedirs(plots_dir, exist_ok=True)

    script_path = os.path.join(os.path.dirname(__file__), "../utils/generate_plots.py")
    if not os.path.exists(script_path):
        script_path = os.path.join(
            os.path.dirname(__file__), "../../src/utils/generate_plots.py"
        )

    if os.path.exists(script_path) and os.path.exists(log_file):
        try:
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--mode",
                    algorithm.lower(),
                    f"--{algorithm.lower()}-log",
                    log_file,
                    "--output",
                    plots_dir,
                    "--window",
                    str(window),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as e:
            print(f"[Plot] Errore generazione grafici (ignorato): {e}")


# ============ STATE PREPROCESSOR ============
def preprocess_state(state_dict):
    """Converte il JSON della mod in un vettore numpy per la Rete Neurale."""
    features = []

    # 1. Player Status
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))

    # 2. Velocity
    features.append(np.clip(state_dict.get("playerVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("playerVelocityY", 0.0) / 20.0, -1.0, 1.0))

    # 3. Terrain Raycasts
    terrain = state_dict.get("terrainInfo", [1.0] * 5)
    if not terrain or len(terrain) < 5:
        terrain = [1.0] * 5
    features.extend(terrain[:5])

    # 4. Boss Info relative
    boss_rel_x = state_dict.get("bossRelativeX", 0.0)
    boss_rel_y = state_dict.get("bossRelativeY", 0.0)
    dist = state_dict.get("distanceToBoss", 50.0) / 50.0
    facing_boss = float(state_dict.get("isFacingBoss", False))

    features.append(np.clip(boss_rel_x / 30.0, -1.0, 1.0))
    features.append(np.clip(boss_rel_y / 30.0, -1.0, 1.0))
    features.append(np.clip(dist, 0.0, 1.0))
    features.append(facing_boss)

    # 5. Boss Status
    features.append(np.clip(state_dict.get("bossVelocityX", 0.0) / 20.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("bossVelocityY", 0.0) / 20.0, -1.0, 1.0))
    features.append(state_dict.get("bossHealth", 100.0) / 100.0)
    features.append(state_dict.get("mantisLordsKilled", 0) / 3.0)

    # 6. Hazards (Il più vicino)
    hazards = state_dict.get("nearbyHazards", [])
    if len(hazards) > 0:
        h = hazards[0]
        features.append(np.clip(h.get("relX", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("relY", 0.0) / 15.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityX", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("velocityY", 0.0) / 20.0, -1.0, 1.0))
        features.append(np.clip(h.get("distance", 15.0) / 15.0, 0.0, 1.0))
    else:
        features.extend([0.0] * 5)

    return np.array(features, dtype=np.float32)


# ============ WAIT FOR ARENA UTILITY ============
def wait_for_arena_ready(env, max_retries=20):
    """
    Aspetta che il gioco carichi l'arena e che il boss sia rilevabile.
    Evita di iniziare il training mentre si è sulla panchina.
    """
    print("⏳ Waiting for Arena & Boss...", end="", flush=True)
    for _ in range(max_retries):
        state = env._receive_state()  # Leggiamo lo stato raw senza fare step
        if state:
            # Se la distanza è < 90, probabilmente siamo in arena (sulla panchina è spesso >100 o nullo)
            # E controlliamo che il player sia vivo
            dist = state.get("distanceToBoss", 100.0)
            hp = state.get("playerHealth", 0)

            if dist < 90.0 and hp > 0:
                print(f" ✓ READY! (Dist: {dist:.1f})")
                return state

        time.sleep(0.5)
        print(".", end="", flush=True)
        # Inviamo un IDLE per tenere sveglio il socket
        env._send_action("IDLE")

    print(" ⚠️ Timeout waiting for arena. Starting anyway (hope for the best).")
    return env.reset()


# ============ MAIN TRAINING LOOP ============
def train_dqn(
    num_episodes=1000,
    max_steps_per_episode=3000,
    batch_size=128,
    learning_rate=1e-5,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=200000,
    checkpoint_dir="checkpoints_dqn_mantis",
    host="localhost",
    port=5555,
    plot_freq=25,
):
    # Creazione cartelle
    ai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir_full = os.path.join(ai_root, checkpoint_dir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    print(f"\n[System] Connecting to Hollow Knight at {host}:{port}...")

    # Inizializzazione Env con retry
    env = None
    while env is None:
        try:
            env = HollowKnightEnv(host=host, port=port, use_reward_shaping=True)
            if not env.connected:
                raise Exception("Socket not connected")
        except Exception as e:
            print(f"[System] Connection failed ({e}). Retrying in 2s...")
            time.sleep(2)
            env = None

    # Aspetta che l'arena sia caricata
    initial_state_dict = wait_for_arena_ready(env)

    # Inizializzazione Agente
    initial_state = preprocess_state(initial_state_dict)
    state_size = len(initial_state)
    action_size = 9  # Le azioni definite nell'Env

    print(f"[System] State Size: {state_size} | Action Size: {action_size}")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000,
    )

    print(f"🔥 DISPOSITIVO: {agent.device}")

    # Caricamento Checkpoint
    latest_path = os.path.join(checkpoint_dir_full, "latest.pth")
    if os.path.exists(latest_path):
        try:
            agent.load(latest_path)
            print("[System] Resumed from 'latest.pth'")
        except Exception:
            print("[System] Could not load checkpoint, starting fresh.")

    # Logging
    log_file = os.path.join(checkpoint_dir_full, "training_log.txt")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(
                "episode,total_reward,steps,global_step,mantis_killed,avg_loss,epsilon\n"
            )

    global_step = agent.steps_done
    best_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"🚀 TRAINING START - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    # --- CICLO EPISODI ---
    for episode in range(num_episodes):
        # Reset Env e attesa 'Ready'
        state_dict = env.reset()
        if not state_dict:  # Se il reset fallisce, riprova
            state_dict = wait_for_arena_ready(env)

        state = preprocess_state(state_dict)
        episode_reward = 0.0
        episode_loss = []

        print(f"🟢 [Episode {episode + 1}] Start...")

        # --- CICLO PASSI (STEPS) ---
        for step in range(max_steps_per_episode):
            global_step += 1

            # 1. Scelta Azione
            epsilon = agent.get_epsilon(epsilon_start, epsilon_end, epsilon_decay)
            action = agent.select_action(state, epsilon=epsilon)

            # 2. Esecuzione (Env Step)
            next_state_dict, reward, done, info = env.step(action)

            # Gestione errore connessione durante episodio
            if "error" in info:
                print(f"❌ Connection lost at step {step}. Ending episode.")
                break

            next_state = preprocess_state(next_state_dict)

            # 3. Training
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.memory) >= batch_size:
                loss = agent.optimize_model(batch_size)
                if loss is not None:
                    episode_loss.append(loss)
                agent.update_target_network(0.005)

            state = next_state

            # --- LOGGING ORDINATO SUL TERMINALE ---

            # A) Rilevamento Eventi Importanti (Danni e Colpi)
            if info.get("damage_taken", 0) > 0:
                # Distinzione tipo danno
                if info.get("spike_damage", False):
                    print(
                        f"   💀 SPUNTONI!  Reward: {reward:.2f} | HP: {next_state_dict.get('playerHealth')}"
                    )
                else:
                    print(
                        f"   🩸 COLPITO DAL BOSS  Reward: {reward:.2f} | HP: {next_state_dict.get('playerHealth')}"
                    )

            elif reward >= 1.0:
                print(
                    f"   ⚔️  COLPO A SEGNO!     Reward: {reward:.2f} | Boss HP: {next_state_dict.get('bossHealth'):.0f}"
                )

            # B) Stato Periodico (Ogni 50 step)
            if step % 50 == 0:
                dist = next_state_dict.get("distanceToBoss", 0.0)
                facing = "SI" if next_state_dict.get("isFacingBoss") else "NO"
                print(
                    f"   ⏱️ Step {step:<4} | Eps: {epsilon:.3f} | Dist: {dist:.1f} | Facing: {facing} | Rew: {episode_reward:.1f}"
                )

            # Fine Episodio
            if done:
                reason = "Morte" if next_state_dict.get("isDead") else "Vittoria!"
                if next_state_dict.get("bossDefeated"):
                    reason = "BOSS SCONFITTO! 🎉"
                print(f"   🏁 FINE: {reason}")
                break

        # --- FINE EPISODIO ---
        agent.episodes_done += 1
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        mantis_killed = next_state_dict.get("mantisLordsKilled", 0)

        # Log su File
        with open(log_file, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{step + 1},{global_step},{mantis_killed},{avg_loss:.4f},{epsilon:.4f}\n"
            )

        # Salvataggio Modelli
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(checkpoint_dir_full, "best_model.pth"))
            print(f"   🌟 NUOVO RECORD! ({episode_reward:.1f}) -> best_model.pth")

        agent.save(latest_path)  # Salva sempre l'ultimo stato

        # Grafici
        if (episode + 1) % plot_freq == 0:
            auto_generate_plots(log_file, checkpoint_dir_full, "DQN", 20, episode + 1)

        print("-" * 60)

    env.close()
    print("Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    # Directory Checkpoint differenziata
    cp_dir = "checkpoints_dqn_mantis"
    if args.instance > 0:
        cp_dir = f"{cp_dir}_instance{args.instance}"

    try:
        train_dqn(num_episodes=args.episodes, port=args.port, checkpoint_dir=cp_dir)
    except KeyboardInterrupt:
        print("\n[System] Training interrotto dall'utente.")
    except Exception as e:
        print(f"\n[System] Errore Critico: {e}")
