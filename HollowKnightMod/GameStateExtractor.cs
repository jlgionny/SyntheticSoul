using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Newtonsoft.Json;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Classe per rappresentare lo stato del gioco da inviare al Python agent.
    /// </summary>
    [Serializable]
    public class GameState
    {
        // PLAYER STATE
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

        // DAMAGE ACCUMULATOR
        public int damageTaken;

        // TERRAIN INFO
        // 0: Floor Distance (sotto)
        // 1: Ceiling Distance (sopra)
        // 2: Wall Ahead (avanti)
        // 3: Wall Behind (dietro)
        // 4: Platform check diagonal
        public float[] terrainInfo;

        // BOSS STATE - FIXED NORMALIZATION
        public float bossX;
        public float bossY;
        public float bossHealth;
        public float distanceToBoss;
        public float bossRelativeX;
        public float bossRelativeY;
        public bool isFacingBoss;
        public bool bossDefeated;

        // MANTIS LORDS TRACKING
        public int mantisLordsKilled;

        // HAZARDS (Projectiles, Enemies, Spikes)
        public List<HazardInfo> nearbyHazards;

        public GameState()
        {
            terrainInfo = new float[5];
            nearbyHazards = new List<HazardInfo>();
            damageTaken = 0;
            mantisLordsKilled = 0;
        }
    }

    [Serializable]
    public class HazardInfo
    {
        public string type; // "projectile", "enemy", "hazard", "spikes", "boomerang"
        public float relX;
        public float relY;
        public float velocityX;
        public float velocityY;
        public float distance; // Distanza dal player
    }

    /// <summary>
    /// Estrae lo stato completo del gioco per l'agente RL.
    /// </summary>
    public class GameStateExtractor
    {
        private const float RAYCAST_DISTANCE = 20f;
        private const float HAZARD_DETECTION_RADIUS = 15f;
        private const int MAX_HAZARDS = 3;

        private LayerMask terrainLayer;
        private LayerMask enemyLayer;
        private LayerMask hazardLayer; // Include sia Terrain che Hazards

        // NUOVO: Traccia il numero di Mantis Lords uccise
        private int prevMantisLordsKilled = 0;

        public GameStateExtractor()
        {
            terrainLayer = LayerMask.GetMask("Terrain");
            enemyLayer = LayerMask.GetMask("Enemies");
            hazardLayer = LayerMask.GetMask("Terrain", "Hazards");

            DesktopLogger.Log($"[Extractor] Initialized with layers - Terrain: {terrainLayer.value}, Hazards: {hazardLayer.value}");
        }

        /// <summary>
        /// NUOVO: Resetta il tracking per un nuovo episodio.
        /// </summary>
        public void ResetTracking()
        {
            prevMantisLordsKilled = 0;
            DesktopLogger.Log("[Extractor] Mantis Lords tracking reset");
        }

        /// <summary>
        /// Estrae lo stato completo del gioco.
        /// </summary>
        public GameState ExtractState()
        {
            var state = new GameState();
            var hero = HeroController.instance;

            if (hero == null)
            {
                DesktopLogger.LogWarning("HeroController is null!");
                return GetDefaultState();
            }

            // PLAYER STATE
            ExtractPlayerState(hero, state);

            // TERRAIN INFO with Hazards/Spikes
            ExtractTerrainInfo();

            // BOSS STATE - FIXED NORMALIZATION
            ExtractBossState(hero, state);

            // HAZARDS (Projectiles, Boomerangs, Spikes)
            ExtractHazards(hero, state);

            return state;
        }

        /// <summary>
        /// Estrae lo stato del player.
        /// </summary>
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
            state.canAttack = !hero.cState.attacking && !hero.cState.recoiling && !hero.cState.dead && !hero.cState.hazardRespawning;
            state.isGrounded = hero.cState.onGround;
            state.isDead = hero.cState.dead;
            state.facingRight = hero.cState.facingRight;
        }

        /// <summary>
        /// Estrae informazioni sul terreno usando raycasts.
        /// </summary>
        private float[] ExtractTerrainInfo()
        {
            float[] terrainInfo = new float[5];
            Vector2 playerPos = HeroController.instance.transform.position;

            // FIX 1: Offset raycast origin di 0.5 unità sopra il player
            Vector2 rayOrigin = new Vector2(playerPos.x, playerPos.y + 0.5f);

            // FIX 2: Usa TUTTI i layer per debug
            LayerMask terrainMask = ~0; // Tutti i layer

            DesktopLogger.Log($"[Raycast] Origin: {rayOrigin}");

            // Ground check - start from ABOVE player
            RaycastHit2D groundHit = Physics2D.Raycast(rayOrigin, Vector2.down, 3f, terrainMask);
            if (groundHit.collider != null && groundHit.distance > 0.1f)
            {
                terrainInfo[0] = Mathf.Clamp01(groundHit.distance / 2f);
                DesktopLogger.Log($"[Ground] Hit: {groundHit.collider.name}, Dist: {groundHit.distance:F2}");
            }
            else
            {
                terrainInfo[0] = 1.0f;
                DesktopLogger.Log("[Ground] No valid hit");
            }

            // Ceiling
            RaycastHit2D ceilingHit = Physics2D.Raycast(rayOrigin, Vector2.up, 3f, terrainMask);
            terrainInfo[1] = (ceilingHit.collider != null && ceilingHit.distance > 0.1f)
                ? Mathf.Clamp01(ceilingHit.distance / 2f) : 1.0f;

            // Wall ahead
            Vector2 facingDir = HeroController.instance.cState.facingRight ? Vector2.right : Vector2.left;
            RaycastHit2D wallHit = Physics2D.Raycast(rayOrigin, facingDir, 5f, terrainMask);
            if (wallHit.collider != null && wallHit.distance > 0.1f)
            {
                terrainInfo[2] = Mathf.Clamp01(wallHit.distance / 3f);
                DesktopLogger.Log($"[Wall] Hit: {wallHit.collider.name}, Dist: {wallHit.distance:F2}");
            }
            else
            {
                terrainInfo[2] = 1.0f;
                DesktopLogger.Log("[Wall] No valid hit");
            }

            // Wall behind
            Vector2 behindDir = HeroController.instance.cState.facingRight ? Vector2.left : Vector2.right;
            RaycastHit2D wallBehindHit = Physics2D.Raycast(rayOrigin, behindDir, 5f, terrainMask);
            terrainInfo[3] = (wallBehindHit.collider != null && wallBehindHit.distance > 0.1f)
                ? Mathf.Clamp01(wallBehindHit.distance / 3f) : 1.0f;

            // Platform
            Vector2 diagDir = new Vector2(facingDir.x, -0.5f).normalized;
            RaycastHit2D diagHit = Physics2D.Raycast(rayOrigin, diagDir, 5f, terrainMask);
            terrainInfo[4] = (diagHit.collider != null && diagHit.distance > 0.1f)
                ? Mathf.Clamp01(diagHit.distance / 3f) : 1.0f;

            return terrainInfo;
        }


        /// <summary>
        /// Esegue un raycast e restituisce la distanza all'ostacolo.
        /// </summary>
        private float RaycastDistance(Vector2 origin, Vector2 direction, float maxDistance, LayerMask layer)
        {
            RaycastHit2D hit = Physics2D.Raycast(origin, direction, maxDistance, layer);
            if (hit.collider != null)
            {
                return hit.distance;
            }
            return maxDistance;
        }

        /// <summary>
        /// Estrae informazioni sul boss con FIXED NORMALIZATION (consistente con wall distance).
        /// </summary>
        private void ExtractBossState(HeroController hero, GameState state)
        {
            GameObject boss = FindBoss();
            if (boss != null)
            {
                var bossPos = boss.transform.position;
                var playerPos = hero.transform.position;

                state.bossX = bossPos.x;
                state.bossY = bossPos.y;

                float dx = bossPos.x - playerPos.x;
                float dy = bossPos.y - playerPos.y;
                state.distanceToBoss = Mathf.Sqrt(dx * dx + dy * dy);

                // FIX CRITICO: Normalizza con Clamp per evitare saturazione
                // Arena Mantis Lords: ~40 unità larghezza, ~30 altezza
                // STESSA SCALA dei raycasts (0-1 normalizzato)
                state.bossRelativeX = Mathf.Clamp(dx / 40.0f, -1.0f, 1.0f);
                state.bossRelativeY = Mathf.Clamp(dy / 30.0f, -1.0f, 1.0f);

                bool facingBoss = (dx > 0 && hero.cState.facingRight) || (dx < 0 && !hero.cState.facingRight);
                state.isFacingBoss = facingBoss;

                var hm = boss.GetComponent<HealthManager>();
                if (hm != null)
                {
                    state.bossHealth = hm.hp;
                    state.bossDefeated = hm.hp <= 0;
                }
                else
                {
                    state.bossHealth = 0f;
                    state.bossDefeated = false;
                }
            }
            else
            {
                state.bossX = 0f;
                state.bossY = 0f;
                state.bossHealth = 0f;
                state.distanceToBoss = 100f;
                state.bossRelativeX = 0f;
                state.bossRelativeY = 0f;
                state.isFacingBoss = false;
                state.bossDefeated = false;
            }

            // TRACKING MANTIS LORDS
            string[] mantisLordNames = new string[]
            {
                "Mantis Lord",      // Prima (Mantis Lord centrale)
                "Mantis Lord S1",   // Seconda (sinistra)
                "Mantis Lord S2"    // Terza (destra)
            };

            int killedCount = 0;
            foreach (string lordName in mantisLordNames)
            {
                GameObject lord = GameObject.Find(lordName);
                if (lord != null)
                {
                    var hm = lord.GetComponent<HealthManager>();
                    // Conta come uccisa SOLO se esiste ed è morta (hp <= 0)
                    if (hm != null && hm.hp <= 0)
                    {
                        killedCount++;
                    }
                }
            }

            state.mantisLordsKilled = killedCount;

            // LOGGA SOLO QUANDO IL NUMERO CAMBIA
            if (killedCount != prevMantisLordsKilled)
            {
                DesktopLogger.Log($"[Mantis Lords] Progress: {killedCount}/3 defeated (previous: {prevMantisLordsKilled})");
                prevMantisLordsKilled = killedCount;
            }
        }

        /// <summary>
        /// Trova il GameObject del boss nella scena.
        /// </summary>
        private GameObject FindBoss()
        {
            DesktopLogger.Log("[FindBoss] Starting boss search...");

            // Lista nomi Mantis Lords
            string[] mantisNames = new string[]
            {
                "Mantis Lord",
                "Mantis Lord S1",
                "Mantis Lord S2"
            };

            foreach (string name in mantisNames)
            {
                GameObject boss = GameObject.Find(name);
                if (boss != null)
                {
                    DesktopLogger.Log($"[FindBoss] Found {name} at {boss.transform.position}");
                    var hm = boss.GetComponent<HealthManager>();
                    if (hm != null)
                    {
                        DesktopLogger.Log($"[FindBoss] {name} HP: {hm.hp}");
                        if (hm.hp > 0)
                        {
                            return boss;
                        }
                    }
                }
                else
                {
                    DesktopLogger.Log($"[FindBoss] {name} NOT FOUND");
                }
            }

            // Fallback: cerca TUTTI i GameObject nella scena
            DesktopLogger.Log("[FindBoss] Fallback: searching all GameObjects...");
            var allObjects = GameObject.FindObjectsOfType<GameObject>();
            DesktopLogger.Log($"[FindBoss] Total objects in scene: {allObjects.Length}");

            // Cerca oggetti con "Mantis" o "Lord" nel nome
            foreach (var obj in allObjects)
            {
                if (obj.name.Contains("Mantis") || obj.name.Contains("Lord"))
                {
                    DesktopLogger.Log($"[FindBoss] Found potential boss: {obj.name} at {obj.transform.position}");
                    var hm = obj.GetComponent<HealthManager>();
                    if (hm != null && hm.hp > 0)
                    {
                        DesktopLogger.Log($"[FindBoss] Using {obj.name} as boss (HP: {hm.hp})");
                        return obj;
                    }
                }
            }

            DesktopLogger.LogError("[FindBoss] NO BOSS FOUND IN SCENE!");
            return null;
        }

        /// <summary>
        /// Estrae informazioni sui pericoli (proiettili, boomerang mantidi, nemici, spike walls).
        /// </summary>
        private void ExtractHazards(HeroController hero, GameState state)
        {
            var playerPos = hero.transform.position;
            List<HazardInfo> hazards = new List<HazardInfo>();

            // 1. BOOMERANG DELLE MANTIDI
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (var obj in allObjects)
            {
                if (obj == null || !obj.activeInHierarchy) continue;

                string name = obj.name.ToLower();
                // Rileva boomerang mantidi
                if (name.Contains("javelin") || name.Contains("boomerang") ||
                    name.Contains("scythe") || name.Contains("mantis") || name.Contains("shot"))
                {
                    float distance = Vector2.Distance(playerPos, obj.transform.position);
                    if (distance <= HAZARD_DETECTION_RADIUS)
                    {
                        var hazard = new HazardInfo
                        {
                            type = "boomerang",
                            relX = obj.transform.position.x - playerPos.x,
                            relY = obj.transform.position.y - playerPos.y,
                            distance = distance
                        };

                        var rb = obj.GetComponent<Rigidbody2D>();
                        if (rb != null)
                        {
                            hazard.velocityX = rb.velocity.x;
                            hazard.velocityY = rb.velocity.y;
                        }

                        hazards.Add(hazard);
                    }
                }
            }

            // 2. PROJECTILES GENERICI (Backup)
            string[] projectileTags = new string[] { "Spell", "Projectile", "Attack" };
            foreach (string tag in projectileTags)
            {
                try
                {
                    GameObject[] projectiles = GameObject.FindGameObjectsWithTag(tag);
                    foreach (var proj in projectiles)
                    {
                        if (proj != null && proj.activeInHierarchy)
                        {
                            float distance = Vector2.Distance(playerPos, proj.transform.position);
                            if (distance <= HAZARD_DETECTION_RADIUS)
                            {
                                var hazard = new HazardInfo
                                {
                                    type = "projectile",
                                    relX = proj.transform.position.x - playerPos.x,
                                    relY = proj.transform.position.y - playerPos.y,
                                    distance = distance
                                };

                                var rb = proj.GetComponent<Rigidbody2D>();
                                if (rb != null)
                                {
                                    hazard.velocityX = rb.velocity.x;
                                    hazard.velocityY = rb.velocity.y;
                                }

                                hazards.Add(hazard);
                            }
                        }
                    }
                }
                catch { }
            }

            // 3. SPIKE WALLS E HAZARDS STATICI
            Vector2[] spikeDirections = new Vector2[]
            {
                Vector2.down, Vector2.up, Vector2.left, Vector2.right,
                new Vector2(-1, -1).normalized, new Vector2(1, -1).normalized,
                new Vector2(-1, 1).normalized, new Vector2(1, 1).normalized
            };

            foreach (var dir in spikeDirections)
            {
                RaycastHit2D hit = Physics2D.Raycast(playerPos, dir, HAZARD_DETECTION_RADIUS, hazardLayer);
                if (hit.collider != null)
                {
                    string hitName = hit.collider.gameObject.name.ToLower();
                    bool isSpike = hitName.Contains("spike") || hitName.Contains("thorn") ||
                                   hitName.Contains("hazard") ||
                                   hit.collider.gameObject.layer == LayerMask.NameToLayer("Hazards");

                    if (isSpike && hit.distance < 10f)
                    {
                        var hazard = new HazardInfo
                        {
                            type = "spikes",
                            relX = hit.point.x - playerPos.x,
                            relY = hit.point.y - playerPos.y,
                            distance = hit.distance,
                            velocityX = 0,
                            velocityY = 0
                        };
                        hazards.Add(hazard);
                    }
                }
            }

            // 4. ENEMIES (Mantis Lords stesse)
            var healthManagers = GameObject.FindObjectsOfType<HealthManager>();
            foreach (var hm in healthManagers)
            {
                if (hm.hp > 0 && hm.hp < 50 && hm.gameObject.activeInHierarchy)
                {
                    float distance = Vector2.Distance(playerPos, hm.transform.position);
                    if (distance <= HAZARD_DETECTION_RADIUS)
                    {
                        var hazard = new HazardInfo
                        {
                            type = "enemy",
                            relX = hm.transform.position.x - playerPos.x,
                            relY = hm.transform.position.y - playerPos.y,
                            distance = distance
                        };

                        var rb = hm.GetComponent<Rigidbody2D>();
                        if (rb != null)
                        {
                            hazard.velocityX = rb.velocity.x;
                            hazard.velocityY = rb.velocity.y;
                        }

                        hazards.Add(hazard);
                    }
                }
            }

            // Ordina per distanza e prendi i più vicini (6 invece di 3 per Mantis Lords)
            hazards = hazards.OrderBy(h => h.distance).ToList();
            state.nearbyHazards = hazards.Take(MAX_HAZARDS * 2).ToList(); // 6 hazard totali

            if (hazards.Count > 0)
            {
                DesktopLogger.Log($"[Hazards] Total detected: {hazards.Count}, sending top {state.nearbyHazards.Count}");
            }
        }

        /// <summary>
        /// Restituisce uno stato di default quando il gioco non è pronto.
        /// </summary>
        private GameState GetDefaultState()
        {
            var state = new GameState();
            state.playerX = 0f;
            state.playerY = 0f;
            state.playerVelocityX = 0f;
            state.playerVelocityY = 0f;
            state.playerHealth = 5;
            state.playerSoul = 0;
            state.canDash = false;
            state.canAttack = true;
            state.isGrounded = true;
            state.hasDoubleJump = false;
            state.isDead = false;
            state.facingRight = true;
            state.damageTaken = 0;

            for (int i = 0; i < 5; i++)
            {
                state.terrainInfo[i] = 1.0f;
            }

            state.bossX = 0f;
            state.bossY = 0f;
            state.bossHealth = 0f;
            state.distanceToBoss = 100f;
            state.bossRelativeX = 0f;
            state.bossRelativeY = 0f;
            state.isFacingBoss = false;
            state.bossDefeated = false;
            state.mantisLordsKilled = 0;

            return state;
        }

        /// <summary>
        /// Serializza lo stato in JSON per inviarlo al Python.
        /// </summary>
        public string SerializeState(GameState state)
        {
            try
            {
                return JsonConvert.SerializeObject(state);
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Failed to serialize GameState: {e.Message}");
                return "{}";
            }
        }
    }
}
