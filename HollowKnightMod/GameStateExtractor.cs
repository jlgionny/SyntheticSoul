using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Newtonsoft.Json;
// Importante per leggere le intenzioni del boss
using HutongGames.PlayMaker;

namespace SyntheticSoulMod
{
    [Serializable]
    public class GameState
    {
        public float playerX;
        public float playerY;
        public float playerVelocityX;
        public float playerVelocityY;
        public int playerHealth;
        public int playerSoul;
        public bool canDash;
        public bool canAttack;
        public bool isGrounded;
        public bool hasDoubleJump;
        public bool isDead;
        public bool facingRight;
        public int damageTaken;
        public float[] terrainInfo;
        public int lastHazardType;

        // Boss Info
        public float bossX;
        public float bossY;
        public float bossHealth;
        public float distanceToBoss;
        public float bossRelativeX;
        public float bossRelativeY;
        public bool isFacingBoss;
        public bool bossDefeated;
        public int mantisLordsKilled;

        // Boss Intent (0=Idle, 1=Dash, 2=Drop, 3=Throw)
        public int bossAction;

        public List<HazardInfo> nearbyHazards;

        public GameState()
        {
            terrainInfo = new float[5];
            nearbyHazards = new List<HazardInfo>();
            damageTaken = 0;
            lastHazardType = 1;
            mantisLordsKilled = 0;
            bossAction = 0;
        }
    }

    [Serializable]
    public class HazardInfo
    {
        public string type;
        public float relX;
        public float relY;
        public float velocityX;
        public float velocityY;
        public float distance;
    }

    public class GameStateExtractor
    {
        private const float HAZARD_DETECTION_RADIUS = 25f; // Aumentato per sicurezza
        private const int MAX_HAZARDS = 3;

        private LayerMask terrainLayer;
        private LayerMask hazardLayer;

        private int prevMantisLordsKilled = 0;

        // MEMORY FIX: Ricordiamo l'ultima vita nota per evitare che vada a 0 se il boss "blinka"
        private float lastKnownBossHealth = 400f;

        public GameStateExtractor()
        {
            terrainLayer = LayerMask.GetMask("Terrain");
            hazardLayer = LayerMask.GetMask("Terrain", "Hazards");
        }

        public void ResetTracking()
        {
            prevMantisLordsKilled = 0;
            lastKnownBossHealth = 400f;
            DesktopLogger.Log("[Extractor] Tracking reset");
        }

        public GameState ExtractState()
        {
            var state = new GameState();
            var hero = HeroController.instance;

            if (hero == null) return GetDefaultState();

            ExtractPlayerState(hero, state);
            ExtractTerrainInfo(hero, state);
            ExtractBossState(hero, state);
            ExtractHazards(hero, state);

            return state;
        }

        private void ExtractPlayerState(HeroController hero, GameState state)
        {
            var pos = hero.transform.position;
            var rb = hero.GetComponent<Rigidbody2D>();

            state.playerX = pos.x;
            state.playerY = pos.y;

            if (rb != null)
            {
                state.playerVelocityX = rb.velocity.x;
                state.playerVelocityY = rb.velocity.y;
            }

            var pd = PlayerData.instance;
            if (pd != null)
            {
                state.playerHealth = pd.health;
                state.playerSoul = pd.MPCharge;
                state.hasDoubleJump = pd.hasDoubleJump;
            }

            state.canDash = !hero.cState.dashing && !hero.cState.backDashing && !hero.cState.shadowDashing && !hero.cState.dashCooldown;
            state.canAttack = !hero.cState.attacking && !hero.cState.recoiling && !hero.cState.dead;
            state.isGrounded = hero.cState.onGround;
            state.isDead = hero.cState.dead;
            state.facingRight = hero.cState.facingRight;
        }

        private void ExtractTerrainInfo(HeroController hero, GameState state)
        {
            Vector2 playerPos = hero.transform.position;
            Vector2 rayOrigin = new Vector2(playerPos.x, playerPos.y + 0.5f);

            // Raycasts normalizzati (0-1)
            RaycastHit2D groundHit = Physics2D.Raycast(rayOrigin, Vector2.down, 5f, terrainLayer);
            state.terrainInfo[0] = (groundHit.collider != null) ? Mathf.Clamp01(groundHit.distance / 3f) : 1.0f;

            RaycastHit2D ceilingHit = Physics2D.Raycast(rayOrigin, Vector2.up, 5f, terrainLayer);
            state.terrainInfo[1] = (ceilingHit.collider != null) ? Mathf.Clamp01(ceilingHit.distance / 3f) : 1.0f;

            Vector2 forward = hero.cState.facingRight ? Vector2.right : Vector2.left;
            Vector2 backward = hero.cState.facingRight ? Vector2.left : Vector2.right;

            RaycastHit2D wallHit = Physics2D.Raycast(rayOrigin, forward, 5f, terrainLayer);
            state.terrainInfo[2] = (wallHit.collider != null) ? Mathf.Clamp01(wallHit.distance / 3f) : 1.0f;

            RaycastHit2D wallBackHit = Physics2D.Raycast(rayOrigin, backward, 5f, terrainLayer);
            state.terrainInfo[3] = (wallBackHit.collider != null) ? Mathf.Clamp01(wallBackHit.distance / 3f) : 1.0f;

            Vector2 diagDir = new Vector2(forward.x, -1f).normalized;
            RaycastHit2D diagHit = Physics2D.Raycast(rayOrigin, diagDir, 5f, terrainLayer);
            state.terrainInfo[4] = (diagHit.collider != null) ? Mathf.Clamp01(diagHit.distance / 3f) : 1.0f;
        }

        private void ExtractBossState(HeroController hero, GameState state)
        {
            // 1. GESTIONE VITA (Somma Totale per Reward)
            // L'IA vede un solo "Boss Gigante" che è la somma delle 3 mantidi.
            float totalHealth = CalculateTotalMantisHealth();

            if (totalHealth > 0)
            {
                state.bossHealth = totalHealth;
                state.bossDefeated = false;
                lastKnownBossHealth = totalHealth;
            }
            else
            {
                // Se la somma è 0, controlliamo se sono morte davvero
                if (CountDeadMantisLords() >= 3)
                {
                    state.bossHealth = 0;
                    state.bossDefeated = true;
                }
                else
                {
                    // Se siamo in transizione (es. una muore, le altre spawnano), mantieni l'ultimo valore
                    state.bossHealth = lastKnownBossHealth;
                }
            }

            // Aggiorna le kill (essenziale per il reward +10)
            state.mantisLordsKilled = CountDeadMantisLords();

            // 2. GESTIONE BERSAGLIO (Dove mirare)
            // Continua a puntare quella attiva più vicina
            GameObject activeBoss = FindActiveMantisLord(hero.transform.position);

            if (activeBoss != null)
            {
                var bossPos = activeBoss.transform.position;
                var playerPos = hero.transform.position;

                state.bossX = bossPos.x;
                state.bossY = bossPos.y;

                float dx = bossPos.x - playerPos.x;
                float dy = bossPos.y - playerPos.y;
                state.distanceToBoss = Mathf.Sqrt(dx * dx + dy * dy);
                state.bossRelativeX = Mathf.Clamp(dx / 40.0f, -1.0f, 1.0f);
                state.bossRelativeY = Mathf.Clamp(dy / 30.0f, -1.0f, 1.0f);

                // Intenzione del boss attivo
                state.bossAction = GetBossIntent(activeBoss);
            }
            else
            {
                // Nessun boss attivo trovato (transizione o vittoria)
                state.distanceToBoss = 100f;
                state.bossAction = 0;
            }
        }

        // --- CALCOLO SOMMA HP (Senza cambiare variabili) ---
        private float CalculateTotalMantisHealth()
        {
            // Nomi interni delle Mantidi
            string[] lords = { "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2" };
            float totalHp = 0;
            bool foundAny = false;

            foreach (var name in lords)
            {
                var obj = GameObject.Find(name);
                if (obj != null)
                {
                    var hm = obj.GetComponent<HealthManager>();
                    // Somma solo se è viva e attiva
                    if (hm != null && hm.hp > 0)
                    {
                        totalHp += (float)hm.hp;
                        foundAny = true;
                    }
                }
            }

            // Se non trova nessuno, ritorna -1 così usiamo la memoria lastKnownBossHealth
            if (!foundAny) return -1f;

            return totalHp;
        }

        // =================================================================
        // LEGGE LA MENTE DEL BOSS (PlayMaker FSM)
        // =================================================================
        private int GetBossIntent(GameObject boss)
        {
            var fsm = boss.GetComponent<PlayMakerFSM>();
            if (fsm == null) return 0; // Unknown

            string activeState = fsm.ActiveStateName.ToLower();

            // Mappatura delle azioni dei Mantis Lords
            // 1: DASH (Attacco a terra)
            if (activeState.Contains("dash") || activeState.Contains("lunge"))
                return 1;

            // 2: DROP (Attacco dall'alto)
            if (activeState.Contains("land") || activeState.Contains("drop") || activeState.Contains("plunge"))
                return 2;

            // 3: THROW (Boomerang dal muro)
            if (activeState.Contains("throw") || activeState.Contains("wall"))
                return 3;

            return 0; // Idle, Movement, or Unknown
        }

        // =================================================================
        // TROVA IL BOSS VERO (Filtrando i Boomerang)
        // =================================================================
        private GameObject FindActiveMantisLord(Vector3 playerPos)
        {
            string[] lords = { "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2" };
            List<GameObject> activeCandidates = new List<GameObject>();

            foreach (var name in lords)
            {
                var obj = GameObject.Find(name);
                if (obj != null)
                {
                    var hm = obj.GetComponent<HealthManager>();
                    // Filtro rigoroso: HP > 0 e nome che non contiene "Shot" o "Scythe"
                    if (hm != null && hm.hp > 0 && !obj.name.ToLower().Contains("shot") && !obj.name.ToLower().Contains("scythe"))
                    {
                        var fsm = obj.GetComponent<PlayMakerFSM>();
                        // Doppio controllo: deve avere la FSM del boss
                        if (fsm != null && fsm.FsmName == "Mantis Lord")
                        {
                            activeCandidates.Add(obj);
                        }
                    }
                }
            }

            if (activeCandidates.Count == 0) return null;
            if (activeCandidates.Count == 1) return activeCandidates[0];

            // FILTRO ANTI-CAMPER: Prendi quello più in basso (Y minore)
            // I boss sui troni sono in alto, quello che combatte è a terra.
            activeCandidates.Sort((a, b) => a.transform.position.y.CompareTo(b.transform.position.y));
            return activeCandidates[0];
        }

        private int CountDeadMantisLords()
        {
            int killed = 0;

            // --- NOMI GAME OBJECT ---
            string boss1Name = "Mantis Lord";      // Fase 1
            string boss2Name = "Mantis Lord S1";   // Fase 2 (Sinistra)
            string boss3Name = "Mantis Lord S2";   // Fase 2 (Destra)

            var boss1 = GameObject.Find(boss1Name);
            var boss2 = GameObject.Find(boss2Name);
            var boss3 = GameObject.Find(boss3Name);

            // --- LOGICA INTELLIGENTE ---

            // 1. Controlliamo se siamo in FASE 2
            // Se le mantidi della Fase 2 esistono e hanno vita, la Fase 1 è vinta per forza.
            bool phase2Started = (boss2 != null && boss2.activeSelf && GetHealth(boss2) > 0) ||
                                 (boss3 != null && boss3.activeSelf && GetHealth(boss3) > 0);

            if (phase2Started)
            {
                // Se siamo in Fase 2, la prima è sicuramente sconfitta (anche se è seduta sul trono)
                killed = 1;

                // Ora contiamo se qualcuna della Fase 2 è morta
                if (IsDead(boss2)) killed++;
                if (IsDead(boss3)) killed++;
            }
            else
            {
                // Se NON siamo in Fase 2, controlliamo solo la prima normale
                if (IsDead(boss1)) killed++;
            }

            return killed;
        }

        // Helper per leggere la vita in modo sicuro
        private int GetHealth(GameObject obj)
        {
            if (obj == null) return 0;
            var hm = obj.GetComponent<HealthManager>();
            return (hm != null) ? hm.hp : 0;
        }

        // Helper per capire se è morto (o sparito)
        private bool IsDead(GameObject obj)
        {
            if (obj == null || !obj.activeSelf) return true; // Sparito = Morto
            return GetHealth(obj) <= 0;
        }
        private void ExtractHazards(HeroController hero, GameState state)
        {
            var playerPos = hero.transform.position;
            var hazards = new List<HazardInfo>();

            Collider2D[] hits = Physics2D.OverlapCircleAll(playerPos, HAZARD_DETECTION_RADIUS);

            foreach (var hit in hits)
            {
                if (hazards.Count >= MAX_HAZARDS * 2) break;

                string name = hit.name.ToLower();
                string hazardType = "unknown";
                bool isSpike = false;
                bool isProjectile = false;
                bool isEnemy = false;

                var rb = hit.GetComponent<Rigidbody2D>();
                var damageHero = hit.GetComponent<DamageHero>();
                bool isStatic = (rb == null) || rb.isKinematic || (rb.velocity.magnitude < 0.1f);

                // SPUNTONI
                if ((damageHero != null && isStatic) || name.Contains("spike") || name.Contains("thorn"))
                {
                    isSpike = true;
                    hazardType = "spike";
                }

                // PROIETTILI (Boomerang)
                if (name.Contains("shot") || name.Contains("scythe") || name.Contains("boomerang") ||
                   (damageHero != null && !isStatic))
                {
                    isProjectile = true;
                    hazardType = "projectile";
                }

                // NEMICI
                if (hit.gameObject.layer == LayerMask.NameToLayer("Enemies"))
                {
                    if (!isSpike && !isProjectile)
                    {
                        if (name.Contains("mantis") && hit.transform.position.y > playerPos.y + 5.0f) continue;

                        isEnemy = true;
                        hazardType = "enemy";
                    }
                }

                if (isSpike || isProjectile || isEnemy)
                {
                    float dist = Vector2.Distance(playerPos, hit.transform.position);

                    var hInfo = new HazardInfo
                    {
                        type = hazardType,
                        relX = hit.transform.position.x - playerPos.x,
                        relY = hit.transform.position.y - playerPos.y,
                        distance = dist,
                        velocityX = rb != null ? rb.velocity.x : 0f,
                        velocityY = rb != null ? rb.velocity.y : 0f
                    };
                    hazards.Add(hInfo);
                }
            }
            state.nearbyHazards = hazards.OrderBy(h => h.distance).Take(MAX_HAZARDS * 2).ToList();
        }

        private GameState GetDefaultState()
        {
            return new GameState
            {
                distanceToBoss = 100f,
                terrainInfo = new float[] { 1, 1, 1, 1, 1 },
                bossAction = 0,
                bossHealth = lastKnownBossHealth
            };
        }
    }
}