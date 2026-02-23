using System;
using System.Collections.Generic;
using UnityEngine;
using HutongGames.PlayMaker;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Sistema avanzato di rilevamento pattern di attacco delle Mantis Lords.
    /// 
    /// ARCHITETTURA:
    /// Ogni mantide ha una FSM PlayMaker chiamata "Mantis Lord" con stati che
    /// corrispondono ai diversi attacchi. Questo detector combina:
    ///   1. Lettura diretta dello stato FSM (intent)
    ///   2. Analisi della velocità/traiettoria (conferma cinematica)
    ///   3. Tracking temporale (durata dell'attacco, fase di wind-up vs active)
    /// 
    /// PATTERN RICONOSCIUTI (IDs):
    ///   0 = IDLE / Sconosciuto
    ///   1 = DASH_HORIZONTAL  (lunge a terra, orizzontale)
    ///   2 = DASH_DIAGONAL    (dash diagonale dall'alto)
    ///   3 = DAGGER_THROW     (boomerang/falce dal muro)
    ///   4 = WALL_THROW       (lancio dal muro laterale)
    ///   5 = LAND_ATTACK      (atterraggio dall'alto, plunge verticale)
    ///   6 = WIND_UP          (preparazione attacco, finestra per reagire)
    ///   7 = RECOVERING       (post-attacco, finestra per punire)
    /// </summary>
    public class MantisAttackDetector
    {
        // ========== STRUTTURA DATI PER SINGOLA MANTIDE ==========
        public class MantisAttackInfo
        {
            public string name;                // "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2"
            public int attackPattern;           // ID del pattern (0-7)
            public string fsmStateName;         // Nome raw dello stato FSM attuale
            public float attackDuration;        // Quanto tempo è in questo pattern (secondi)
            public float velocityX;             // Velocità X attuale
            public float velocityY;             // Velocità Y attuale
            public float posX;                  // Posizione X
            public float posY;                  // Posizione Y
            public bool isWindingUp;            // True se sta preparando un attacco
            public bool isActiveAttack;         // True se l'attacco è in fase attiva (hitbox attiva)
            public bool isRecovering;           // True se in recovery post-attacco
            public float relativeX;             // Posizione relativa al player (normalizzata)
            public float relativeY;
            public int directionToPlayer;       // -1 = player a sinistra, +1 = player a destra
            public float distanceToPlayer;
            public bool isAlive;

            public MantisAttackInfo()
            {
                attackPattern = 0;
                fsmStateName = "";
                isAlive = false;
            }
        }

        // ========== STATO INTERNO PER TRACKING TEMPORALE ==========
        private class MantisTracker
        {
            public string lastFsmState = "";
            public float stateStartTime = 0f;
            public int lastPattern = 0;
            public Vector3 lastPosition = Vector3.zero;
            public float lastUpdateTime = 0f;
        }

        private Dictionary<string, MantisTracker> trackers = new Dictionary<string, MantisTracker>();
        private static readonly string[] MANTIS_NAMES = { "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2" };

        // ========== MAPPATURA STATI FSM -> PATTERN ==========
        // Basata sull'analisi della FSM "Mantis Lord" di Hollow Knight.
        // Gli stati FSM effettivi possono variare leggermente, quindi usiamo substring matching.

        // Stati di WIND-UP (preparazione, l'agente può reagire)
        private static readonly string[] WIND_UP_STATES = {
            "antic",        // Preparazione generica
            "aim",          // Mira il giocatore
            "ready",        // Pronto a partire
            "set",          // Posizionamento
            "telegraph",    // Telegrafing dell'attacco
        };

        // Stati di RECOVERY (post-attacco, finestra di punizione)
        private static readonly string[] RECOVERY_STATES = {
            "recover",
            "land",         // Atterraggio post-dash (quando non è l'attacco stesso)
            "return",       // Ritorno alla posizione
            "rest",
            "cooldown",
            "pause",
        };

        // Stati IDLE/non-combat
        private static readonly string[] IDLE_STATES = {
            "idle",
            "wait",
            "sleep",
            "roar",
            "intro",
            "init",
            "start",
            "away",         // Fuori dall'arena
            "throne",       // Seduta sul trono
        };

        // ========== API PUBBLICA ==========

        /// <summary>
        /// Analizza tutte le mantidi attive e restituisce informazioni sui loro pattern.
        /// Chiamare ogni frame (o al rate dell'UPDATE_INTERVAL della mod).
        /// </summary>
        public List<MantisAttackInfo> DetectAllPatterns(Vector3 playerPos)
        {
            var results = new List<MantisAttackInfo>();

            foreach (var mantisName in MANTIS_NAMES)
            {
                var info = DetectPattern(mantisName, playerPos);
                if (info != null)
                {
                    results.Add(info);
                }
            }

            return results;
        }

        /// <summary>
        /// Restituisce il pattern della mantide attiva più pericolosa 
        /// (quella più vicina che sta attaccando).
        /// </summary>
        public MantisAttackInfo GetMostDangerousMantis(Vector3 playerPos)
        {
            var all = DetectAllPatterns(playerPos);
            MantisAttackInfo mostDangerous = null;
            float minDangerDist = float.MaxValue;

            foreach (var info in all)
            {
                if (!info.isAlive) continue;

                // Priorità: attacco attivo > wind-up > idle
                float dangerScore = info.distanceToPlayer;
                if (info.isActiveAttack) dangerScore -= 100f;  // Massima priorità
                else if (info.isWindingUp) dangerScore -= 50f;

                if (dangerScore < minDangerDist)
                {
                    minDangerDist = dangerScore;
                    mostDangerous = info;
                }
            }

            return mostDangerous;
        }

        public void Reset()
        {
            trackers.Clear();
        }

        // ========== LOGICA DI DETECTION ==========

        private MantisAttackInfo DetectPattern(string mantisName, Vector3 playerPos)
        {
            var obj = GameObject.Find(mantisName);
            if (obj == null || !obj.activeSelf) return null;

            var hm = obj.GetComponent<HealthManager>();
            if (hm == null) return null;

            var info = new MantisAttackInfo();
            info.name = mantisName;
            info.isAlive = hm.hp > 0;
            info.posX = obj.transform.position.x;
            info.posY = obj.transform.position.y;

            // Posizione relativa al player
            float dx = info.posX - playerPos.x;
            float dy = info.posY - playerPos.y;
            info.relativeX = Mathf.Clamp(dx / 40f, -1f, 1f);
            info.relativeY = Mathf.Clamp(dy / 30f, -1f, 1f);
            info.distanceToPlayer = Mathf.Sqrt(dx * dx + dy * dy);
            info.directionToPlayer = (dx > 0) ? -1 : 1;

            // Velocità dalla Rigidbody2D
            var rb = obj.GetComponent<Rigidbody2D>();
            if (rb != null)
            {
                info.velocityX = rb.velocity.x;
                info.velocityY = rb.velocity.y;
            }

            // Tracking temporale
            if (!trackers.ContainsKey(mantisName))
                trackers[mantisName] = new MantisTracker();
            var tracker = trackers[mantisName];

            // Leggi stato FSM
            var fsm = GetMantisFSM(obj);
            string currentFsmState = (fsm != null) ? fsm.ActiveStateName : "";
            info.fsmStateName = currentFsmState;

            // Controlla se lo stato è cambiato
            float now = Time.time;
            if (currentFsmState != tracker.lastFsmState)
            {
                tracker.lastFsmState = currentFsmState;
                tracker.stateStartTime = now;
            }
            info.attackDuration = now - tracker.stateStartTime;

            // Calcola velocità derivata (fallback se rb non disponibile)
            if (tracker.lastUpdateTime > 0 && rb == null)
            {
                float dt = now - tracker.lastUpdateTime;
                if (dt > 0.001f)
                {
                    info.velocityX = (obj.transform.position.x - tracker.lastPosition.x) / dt;
                    info.velocityY = (obj.transform.position.y - tracker.lastPosition.y) / dt;
                }
            }
            tracker.lastPosition = obj.transform.position;
            tracker.lastUpdateTime = now;

            // ===== CLASSIFICAZIONE PATTERN =====
            info.attackPattern = ClassifyAttackPattern(currentFsmState, info.velocityX, info.velocityY, info.posY, playerPos.y);

            // Fase dell'attacco
            info.isWindingUp = IsWindUpState(currentFsmState);
            info.isRecovering = IsRecoveryState(currentFsmState);
            info.isActiveAttack = info.attackPattern >= 1 && info.attackPattern <= 5
                                   && !info.isWindingUp && !info.isRecovering;

            tracker.lastPattern = info.attackPattern;

            return info;
        }

        private int ClassifyAttackPattern(string fsmState, float velX, float velY, float mantisY, float playerY)
        {
            if (string.IsNullOrEmpty(fsmState)) return 0;
            string state = fsmState.ToLower();

            // 1. IDLE / Non-combat
            foreach (var idle in IDLE_STATES)
            {
                if (state.Contains(idle)) return 0;
            }

            // 2. WIND-UP (preparazione) - Pattern 6
            if (IsWindUpState(fsmState))
                return 6;

            // 3. RECOVERY (post-attacco) - Pattern 7
            // Nota: "land" può essere sia recovery che attacco.
            // Se la velocità Y è alta e negativa, è un plunge attack, non recovery.
            if (IsRecoveryState(fsmState) && !(state.Contains("land") && velY < -5f))
                return 7;

            // 4. DASH ORIZZONTALE (Pattern 1)
            // La mantide si lancia orizzontalmente a velocità elevata
            if (state.Contains("dash") || state.Contains("lunge") || state.Contains("charge"))
            {
                // Conferma cinematica: velocità X elevata, Y bassa
                if (Mathf.Abs(velX) > 5f && Mathf.Abs(velY) < 3f)
                    return 1; // DASH_HORIZONTAL confermato

                // Se ha anche componente Y significativa -> diagonale
                if (Mathf.Abs(velX) > 3f && Mathf.Abs(velY) > 3f)
                    return 2; // DASH_DIAGONAL

                // Default: è un dash ma non abbiamo conferma cinematica ancora
                // (potrebbe essere nel primissimo frame)
                return 1;
            }

            // 5. DASH DIAGONALE (Pattern 2)
            // La mantide attacca in diagonale, tipicamente dall'alto
            if (state.Contains("diag") || state.Contains("swoop"))
            {
                return 2;
            }

            // 6. LANCIO PROIETTILE / BOOMERANG (Pattern 3)
            if (state.Contains("throw") || state.Contains("scythe") || state.Contains("boomerang") || state.Contains("shot"))
            {
                return 3; // DAGGER_THROW
            }

            // 7. ATTACCO DAL MURO (Pattern 4)
            // La mantide si attacca al muro e lancia un proiettile
            if (state.Contains("wall"))
            {
                // Se contiene anche "throw", è un wall throw
                if (state.Contains("throw") || state.Contains("shot"))
                    return 4; // WALL_THROW
                // Altrimenti potrebbe essere posizionamento sul muro
                return 4;
            }

            // 8. ATTACCO DALL'ALTO / PLUNGE (Pattern 5)
            // La mantide cade dall'alto sulla posizione del giocatore
            if (state.Contains("drop") || state.Contains("plunge") || state.Contains("slam"))
            {
                return 5; // LAND_ATTACK
            }

            // 9. "land" ambiguo: se viene dall'alto con velocità Y negativa alta
            if (state.Contains("land"))
            {
                if (velY < -5f && mantisY > playerY + 2f)
                    return 5; // È un plunge attack
                else
                    return 7; // Recovery dopo atterraggio
            }

            // 10. Fallback cinematico: se la FSM non matcha ma la velocità è alta
            float speed = Mathf.Sqrt(velX * velX + velY * velY);
            if (speed > 15f)
            {
                // Qualcosa si sta muovendo veloce -> probabilmente un attacco
                if (Mathf.Abs(velX) > Mathf.Abs(velY) * 2f)
                    return 1; // Dash orizzontale
                if (Mathf.Abs(velY) > Mathf.Abs(velX) * 2f)
                    return 5; // Plunge verticale
                return 2;     // Diagonale
            }

            return 0; // Idle o sconosciuto
        }

        private bool IsWindUpState(string fsmState)
        {
            if (string.IsNullOrEmpty(fsmState)) return false;
            string state = fsmState.ToLower();
            foreach (var wu in WIND_UP_STATES)
            {
                if (state.Contains(wu)) return true;
            }
            return false;
        }

        private bool IsRecoveryState(string fsmState)
        {
            if (string.IsNullOrEmpty(fsmState)) return false;
            string state = fsmState.ToLower();
            foreach (var rc in RECOVERY_STATES)
            {
                if (state.Contains(rc)) return true;
            }
            return false;
        }

        private PlayMakerFSM GetMantisFSM(GameObject mantis)
        {
            PlayMakerFSM[] fsms = mantis.GetComponents<PlayMakerFSM>();
            foreach (var fsm in fsms)
            {
                if (fsm.FsmName == "Mantis Lord")
                    return fsm;
            }
            // Fallback: qualsiasi FSM di controllo
            foreach (var fsm in fsms)
            {
                string name = fsm.FsmName.ToLower();
                if (name.Contains("control") || name.Contains("attack"))
                    return fsm;
            }
            return fsms.Length > 0 ? fsms[0] : null;
        }
    }
}
