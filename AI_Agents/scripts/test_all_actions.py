import os
import sys
import time

# Setup path per importare hollow_knight_env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.env.hollow_knight_env import HollowKnightEnv


def test_full_abilities():
    print("=" * 60)
    print("🛠️  TEST COMPLETO ABILITÀ (AUDIO & ACTIONS CHECK)")
    print("=" * 60)

    env = HollowKnightEnv(host="localhost", port=5555)
    env.reset()

    # Aspettiamo un attimo che il gioco si stabilizzi
    time.sleep(1)

    # ACTIONS MAP (Reference):
    # 0: Left, 1: Right, 2: Up, 3: Down, 4: Jump, 5: Attack, 6: Dash, 7: Spell, 8: IDLE

    # Sequenza di test: (Nome, ID, Frame di durata, Descrizione)
    # Nota: 60 frame ~= 1 secondo a velocità normale (ma tu hai 2x quindi è 0.5s)
    sequence = [
        ("🛑 CHECK INIZIALE IDLE", 8, 30, "Verifica se ci sono suoni o inchini"),
        ("➡️ CAMMINATA DESTRA", 1, 60, "Movimento base"),
        ("🛑 STOP (Check Audio)", 8, 40, "Il suono dei passi deve fermarsi SUBITO"),
        ("⬅️ CAMMINATA SINISTRA", 0, 60, "Movimento base"),
        ("🛑 STOP (Check Audio)", 8, 40, "Check silenzio"),
        ("🦘 SALTO (Tap)", 4, 10, "Salto piccolo"),
        ("🛑 ATTERRAGGIO", 8, 30, "Attesa"),
        ("🦘 SALTO (Lungo)", 4, 30, "Salto alto"),
        ("🛑 ATTERRAGGIO", 8, 30, "Attesa"),
        ("⚔️ ATTACCO (Fermo)", 5, 5, "Fendente da fermo"),
        ("🛑 RECOIL", 8, 20, "Recupero"),
        ("💨 DASH (Destra)", 6, 5, "Scatto rapido (Richiede Cooldown)"),
        ("🛑 RECOIL", 8, 40, "Attesa Dash"),
        ("🔥 SPELL / CAST", 7, 5, "Richiede ANIMA (Soul)"),
        ("🛑 RECOIL", 8, 40, "Attesa"),
        ("👀 GUARDA SU", 2, 50, "Deve guardare in alto"),
        ("🛑 STOP", 8, 20, "Torna normale"),
        ("👀 GUARDA GIU", 3, 50, "Deve accucciarsi"),
        ("🛑 STOP", 8, 30, "Torna in piedi (NO INCHINO BUG)"),
    ]

    try:
        print("Inizio Test tra 3 secondi...")
        time.sleep(3)

        for name, action_id, duration, desc in sequence:
            print(f"👉 [{name}]: {desc}")

            for _ in range(duration):
                env.step(action_id)
                # Niente sleep qui, usiamo il tick del gioco

        print("\n✅ Test Completato. Verifica visiva/uditiva richiesta.")

    except KeyboardInterrupt:
        print("\nTest interrotto.")
    finally:
        env.close()


if __name__ == "__main__":
    test_full_abilities()
