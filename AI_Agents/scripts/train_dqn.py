import os
import sys
import subprocess
import math
import numpy as np
from datetime import datetime
import torch

# Setup dei path per importare i moduli src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agents.dqn_agent import DQNAgent
from src.env.hollow_knight_env import HollowKnightEnv


# ============ AUTO PLOT GENERATOR ============
def auto_generate_plots(
    log_file, checkpoint_dir, algorithm="DQN", window=20, current_episode=0
):
    """Genera automaticamente grafici in sottocartelle organizzate."""
    episode_folder = f"episode_{current_episode}"
    plots_dir = os.path.join(checkpoint_dir, "plots", episode_folder)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(log_file):
        print(f"[Auto Plot] Warning: Log file non trovato: {log_file}")
        return

    # Cerca lo script di plotting in diverse posizioni comuni
    script_path = os.path.join(os.path.dirname(__file__), "../utils/generate_plots.py")
    if not os.path.exists(script_path):
        script_path = os.path.join(
            os.path.dirname(__file__), "../../src/utils/generate_plots.py"
        )

    if not os.path.exists(script_path):
        print(
            "[Auto Plot] Warning: Script generate_plots.py non trovato. Salto generazione grafici."
        )
        return

    try:
        # print("\n[Auto Plot] Generazione grafici in corso...")
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
        # print(f"[Auto Plot] ✓ Grafici salvati in: {plots_dir}")
    except Exception as e:
        print(f"[Auto Plot] ✗ Errore non bloccante: {e}")


def preprocess_state(state_dict):
    """
    Converte il JSON della Mod in un array NumPy normalizzato per la Rete Neurale.
    Include gestione robusta di liste vuote e normalizzazione.
    """
    features = []

    # 1. PLAYER FEATURES (10)
    player_x = state_dict.get("playerX", 0.0)
    player_y = state_dict.get("playerY", 0.0)

    # Normalizzazione basata su coordinate approssimative dell'arena Mantis
    features.append(player_x / 40.0)
    features.append(player_y / 30.0)

    features.append(np.clip(state_dict.get("playerVelocityX", 0.0) / 15.0, -1.0, 1.0))
    features.append(np.clip(state_dict.get("playerVelocityY", 0.0) / 15.0, -1.0, 1.0))
    features.append(state_dict.get("playerHealth", 0) / 10.0)
    features.append(state_dict.get("playerSoul", 0) / 100.0)

    # Booleani convertiti in float
    features.append(float(state_dict.get("canDash", False)))
    features.append(float(state_dict.get("canAttack", False)))
    features.append(float(state_dict.get("isGrounded", False)))
    features.append(float(state_dict.get("hasDoubleJump", False)))

    # 2. TERRAIN INFO (5) - Muri e Soffitti
    terrain_info = state_dict.get("terrainInfo", [1.0, 1.0, 1.0, 1.0, 1.0])
    if not terrain_info or len(terrain_info) < 5:
        terrain_info = [1.0] * 5
    features.extend(terrain_info[:5])

    # 3. BOSS FEATURES (7)
    boss_x = state_dict.get("bossX", 0.0)
    boss_y = state_dict.get("bossY", 0.0)

    # Posizione relativa (più importante delle coordinate assolute)
    boss_relative_x = (boss_x - player_x) / 40.0
    boss_relative_y = (boss_y - player_y) / 30.0
    features.append(boss_relative_x)
    features.append(boss_relative_y)

    features.append(state_dict.get("bossHealth", 0) / 1000.0)

    distance_to_boss = state_dict.get("distanceToBoss", 50.0)
    features.append(np.clip(distance_to_boss / 50.0, 0.0, 1.0))

    # Angolo verso il boss
    angle_to_boss = math.atan2(boss_relative_y, boss_relative_x) / math.pi
    features.append(angle_to_boss)

    features.append(float(state_dict.get("isFacingBoss", False)))

    boss_vel_x = state_dict.get("bossVelocityX", 0.0)
    features.append(np.clip(boss_vel_x / 15.0, -1.0, 1.0))

    # Salviamo dati grezzi nel dict per debug (non usati dalla rete)
    state_dict["bossRelativeX"] = boss_relative_x
    state_dict["bossRelativeY"] = boss_relative_y

    # 4. HAZARDS (Top 3 pericoli più vicini)
    hazards = state_dict.get("nearbyHazards", [])
    if hazards:
        hazards_with_dist = []
        for h in hazards:
            rel_x = h.get("relX", 0.0)
            rel_y = h.get("relY", 0.0)
            dist = math.sqrt(rel_x**2 + rel_y**2)
            hazards_with_dist.append((dist, h))

        # Ordina per distanza crescente (il più vicino è l'indice 0)
        hazards_with_dist.sort(key=lambda x: x[0])
        sorted_hazards = [h for _, h in hazards_with_dist[:3]]
    else:
        sorted_hazards = []

    # Riempiamo sempre 3 slot hazard (zero-padding se mancano)
    for i in range(3):
        if i < len(sorted_hazards):
            h = sorted_hazards[i]
            rel_x = h.get("relX", 0.0) / 30.0
            rel_y = h.get("relY", 0.0) / 30.0
            features.append(np.clip(rel_x, -1.0, 1.0))
            features.append(np.clip(rel_y, -1.0, 1.0))

            # Velocità relativa hazard vs player (utile per schivare boomerangs)
            h_vx = h.get("velocityX", 0.0)
            p_vx = state_dict.get("playerVelocityX", 0.0)
            rel_vx = (h_vx - p_vx) / 20.0
            features.append(np.clip(rel_vx, -1.0, 1.0))
        else:
            # Padding: se non c'è pericolo, mettiamo 0
            features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def train_dqn(
    num_episodes=1000,
    max_steps_per_episode=3000,  # Ridotto leggermente per iterazioni più rapide
    batch_size=128,  # AUMENTATO per stabilità (era 64)
    learning_rate=1e-5,  # RIDOTTO per evitare Exploding Loss (era 1e-4)
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=100000,
    checkpoint_dir="checkpoints_dqn_mantis",
    host="localhost",
    port=5555,
    plot_freq=50,
):
    """
    Main training loop con Soft Updates e Hyperparameters stabilizzati.
    """

    ai_agents_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir_full = os.path.join(ai_agents_root, checkpoint_dir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    print(f"[Train] Connecting to Hollow Knight at {host}:{port}...")

    # Environment Initialization
    env = HollowKnightEnv(host=host, port=port, use_reward_shaping=True)

    # Inizializzazione dimensioni stato
    initial_state = env.reset()
    state_array = preprocess_state(initial_state)
    state_size = len(state_array)
    action_size = 8

    print(f"[Train] State size: {state_size}, Action size: {action_size}")
    print("[Train] Mode: SOFT UPDATES + LOW LR (Stabilization Fix)")

    # Agent Initialization
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=100000,
    )

    # --- GPU CHECK ---
    print(f"\n{'='*40}")
    print(f"🔥 DISPOSITIVO: {agent.device}")
    if str(agent.device) == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*40}\n")

    # --- CARICAMENTO CHECKPOINT ---
    # Nota: Se stai ripartendo da zero per fixare la loss, cancella la cartella checkpoints manualmente prima!
    latest_checkpoint = os.path.join(checkpoint_dir_full, "latest.pth")
    if os.path.exists(latest_checkpoint):
        try:
            agent.load(latest_checkpoint)
            print("[Train] Resumed from LATEST checkpoint")
        except Exception as e:
            print(f"[Train] Error loading latest (starting fresh): {e}")

    # Log setup
    log_file = os.path.join(checkpoint_dir_full, "training_log.txt")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(
                "episode,total_reward,steps,global_step,mantis_killed,avg_loss,epsilon\n"
            )

    global_step = agent.steps_done  # Sync global step con l'agente caricato
    best_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for episode in range(num_episodes):
        # Reset Environment
        state_dict = env.reset()
        state = preprocess_state(state_dict)

        episode_reward = 0.0
        episode_loss = []
        total_damage_taken = 0

        print(f"\n[Episode {episode + 1}/{num_episodes}] Running...")

        for step in range(max_steps_per_episode):
            global_step += 1

            # 1. Action Selection
            epsilon = agent.get_epsilon(epsilon_start, epsilon_end, epsilon_decay)
            action = agent.select_action(state, epsilon=epsilon)

            # --- DEBUG BLOCK ---
            action_name = env.ACTIONS.get(action, "UNKNOWN")
            print(
                f"DEBUG: Eps: {epsilon:.2f} | Action Num: {action} | Action Name: {action_name}"
            )
            # -------------------
            # Se siamo all'inizio (Epsilon alto), rallenta per farci leggere
            # if epsilon > 0.5:
            #    time.sleep(0.2) # <--- 200ms di pausa ad ogni frame
            # --------------------------------
            # --- MODIFICA VISIVA ---
            # Stampa solo se lo step è multiplo di 60 (circa 1 secondo di gioco)
            if step % 60 == 0:
                action_name = env.ACTIONS.get(action, "UNKNOWN")
                print(f"\n👀 GUARDA ORA -> L'IA ha scelto: {action_name}")
                print("   (Il personaggio nel gioco deve fare QUESTA azione e basta)")
            # -----------------------# --- MODIFICA VISIVA ---

            # 2. Environment Step
            next_state_dict, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state_dict)

            # Accumulo metriche
            episode_reward += reward
            damage_this_step = (
                info.get("damage_taken", 0) if "damage_taken" in info else 0
            )
            total_damage_taken += damage_this_step

            # 3. Store in Buffer
            agent.store_transition(state, action, reward, next_state, done)

            # 4. TRAINING & SOFT UPDATE (Il cuore della stabilizzazione)
            if len(agent.memory) >= batch_size:
                loss = agent.optimize_model(batch_size=batch_size)
                if loss is not None:
                    episode_loss.append(loss)

                # --- SOFT UPDATE ---
                # Aggiorniamo la target net pochissimo (0.5%) ad ogni step.
                # Questo previene i salti improvvisi della Loss.
                agent.update_target_network(tau=0.005)

            state = next_state

            # 5. Debug Console (Ogni 50 step per non spammare)
            if step % 50 == 0:
                # Controllo Muri
                terrain = next_state_dict.get("terrainInfo", [1.0] * 5)
                # terrain[2]=ahead, terrain[3]=behind
                wall_ahead = terrain[2] if len(terrain) > 2 else 1.0
                wall_behind = terrain[3] if len(terrain) > 3 else 1.0

                wall_status = "LIBERO"
                if wall_ahead < 1.0 or wall_behind < 1.0:
                    wall_status = "⚠️ MURO"

                # Posizione Boss
                # boss_rel_y = next_state_dict.get("bossRelativeY", 0.0)
                dist_boss = next_state_dict.get("distanceToBoss", 0.0)

                print(
                    f"  [{step}] R:{episode_reward:.1f} | L:{np.mean(episode_loss[-10:]) if episode_loss else 0:.3f} | "
                    f"Wall: {wall_status} | BossDist: {dist_boss:.1f}"
                )

            if done:
                reason = "Dead" if next_state_dict.get("isDead") else "Boss Defeated"
                print(f"  [End] Reason: {reason}")
                break

        agent.episodes_done += 1

        # Statistiche Fine Episodio
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        current_mantis_killed = next_state_dict.get("mantisLordsKilled", 0)

        print(
            f"[Summary Ep {episode + 1}] Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} | Mantis: {current_mantis_killed}"
        )

        # Salvataggio Log
        with open(log_file, "a") as f:
            f.write(
                f"{episode + 1},{episode_reward:.2f},{step + 1},{global_step},{current_mantis_killed},{avg_loss:.4f},{epsilon:.4f}\n"
            )

        # Salvataggio Best Model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir_full, "best_model.pth")
            agent.save(best_path)
            print(f"  ★ NEW BEST! Saved to {best_path}")

        # Salvataggio Latest
        agent.save(latest_checkpoint)

        # Plotting Automatico
        if (episode + 1) % plot_freq == 0:
            auto_generate_plots(log_file, checkpoint_dir_full, "DQN", 20, episode + 1)

    env.close()
    print("Training Finished.")


if __name__ == "__main__":
    # Nota: I parametri sono ora definiti dentro la funzione train_dqn come default.
    # Puoi sovrascriverli qui se vuoi, ma i default sono già ottimizzati.
    try:
        train_dqn()
    except KeyboardInterrupt:
        print("\n[Train] Interrupted by user. Saving latest...")
    except Exception as e:
        print(f"\n[Train] Critical Error: {e}")
        import traceback

        traceback.print_exc()
