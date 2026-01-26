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
        // ============ PLAYER STATE ============
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

        // ============ DAMAGE ACCUMULATOR ============
        public int damageTaken;

        // ============ TERRAIN INFO ============
        // [0] = Floor Distance (sotto)
        // [1] = Gap Ahead (avanti-basso)
        // [2] = Wall Ahead (avanti)
        // [3] = Ceiling Ahead (avanti-alto)
        // [4] = Ceiling Distance (sopra)
        public float[] terrainInfo;

        // ============ BOSS STATE (ENHANCED) ============
        public float bossX;
        public float bossY;
        public float bossHealth;
        public float distanceToBoss;
        public float bossRelativeX;
        public float bossRelativeY;
        public bool isFacingBoss;
        public bool bossDefeated;

        // ============ MANTIS LORDS TRACKING ============
        public int mantisLordsKilled;

        // ============ HAZARDS (Projectiles, Enemies, Spikes) ============
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

        // NUOVO: Traccia il numero di Mantis Lords uccise per loggare solo i cambiamenti
        private int prevMantisLordsKilled = 0;

        public GameStateExtractor()
        {
            terrainLayer = LayerMask.GetMask("Terrain");
            enemyLayer = LayerMask.GetMask("Enemies");
            hazardLayer = LayerMask.GetMask("Terrain", "Hazards");
            DesktopLogger.Log($"[Extractor] Initialized with layers - Terrain: {terrainLayer.value}, Hazards: {hazardLayer.value}");
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

            // ============ PLAYER STATE ============
            ExtractPlayerState(hero, state);

            // ============ TERRAIN INFO (with Hazards/Spikes) ============
            ExtractTerrainInfo(hero, state);

            // ============ BOSS STATE (ENHANCED) ============
            ExtractBossState(hero, state);

            // ============ HAZARDS (Projectiles, Boomerangs, Spikes) ============
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

            state.canDash = !hero.cState.dashing &&
                           !hero.cState.backDashing &&
                           !hero.cState.shadowDashing &&
                           !hero.cState.dashCooldown;

            state.canAttack = !hero.cState.attacking &&
                             !hero.cState.recoiling &&
                             !hero.cState.dead &&
                             !hero.cState.hazardRespawning;

            state.isGrounded = hero.cState.onGround;
            state.isDead = hero.cState.dead;
            state.facingRight = hero.cState.facingRight;
        }

        /// <summary>
        /// Estrae informazioni sul terreno E hazards fissi (spikes) usando raycasts.
        /// </summary>
        private void ExtractTerrainInfo(HeroController hero, GameState state)
        {
            var pos = hero.transform.position;
            bool facingRight = hero.cState.facingRight;
            float direction = facingRight ? 1f : -1f;

            // [0] Floor Distance (sotto) - usa hazardLayer per rilevare anche spikes
            state.terrainInfo[0] = RaycastDistance(pos, Vector2.down, RAYCAST_DISTANCE, hazardLayer);

            // [1] Gap Ahead (avanti-basso, diagonale 45°)
            Vector2 gapDir = new Vector2(direction, -1f).normalized;
            state.terrainInfo[1] = RaycastDistance(pos, gapDir, RAYCAST_DISTANCE, hazardLayer);

            // [2] Wall Ahead (avanti, orizzontale) - include spike walls
            Vector2 wallDir = new Vector2(direction, 0f);
            state.terrainInfo[2] = RaycastDistance(pos, wallDir, RAYCAST_DISTANCE, hazardLayer);

            // [3] Ceiling Ahead (avanti-alto, diagonale 45°)
            Vector2 ceilingAheadDir = new Vector2(direction, 1f).normalized;
            state.terrainInfo[3] = RaycastDistance(pos, ceilingAheadDir, RAYCAST_DISTANCE, hazardLayer);

            // [4] Ceiling Distance (sopra)
            state.terrainInfo[4] = RaycastDistance(pos, Vector2.up, RAYCAST_DISTANCE, hazardLayer);

            // Normalizza distanze: 0 = molto vicino, 1 = molto lontano
            for (int i = 0; i < state.terrainInfo.Length; i++)
            {
                state.terrainInfo[i] = Mathf.Clamp01(state.terrainInfo[i] / RAYCAST_DISTANCE);
            }
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
        /// Estrae informazioni sul boss con ENHANCED DIRECTIONAL AWARENESS e Mantis Lords Tracking.
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
                state.bossRelativeX = dx / 20.0f;
                state.bossRelativeY = dy / 20.0f;

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

            // ============ TRACKING MANTIS LORDS (OTTIMIZZATO) ============
            string[] mantisLordNames = new string[]
            {
                "Mantis Lord",      // Prima Mantis Lord (centrale)
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
                    // Se non ha HealthManager o hp <= 0, è morta
                    if (hm == null || hm.hp <= 0)
                    {
                        killedCount++;
                    }
                }
                else
                {
                    // GameObject non esiste = distrutto (ucciso)
                    killedCount++;
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
            string[] bossNames = new string[]
            {
                "Hornet Boss",
                "Hornet",
                "Mantis Lord",
                "False Knight",
                "Mawlek Body",
                "Mega Zombie Beam Miner (1)",
                "Zombie Beam Miner Rematch",
                "Mage Knight",
                "Dream Mage",
                "Ghost Warrior Hu",
                "Ghost Warrior Galien",
                "Ghost Warrior Marmu",
                "Ghost Warrior Xero",
                "Ghost Warrior Markoth",
                "Ghost Warrior No Eyes",
                "Ghost Warrior Gorb",
                "Jar Collector",
                "Dung Defender",
                "White Defender",
                "Lost Kin",
                "Infected Knight",
                "Mantis Traitor Lord",
                "Hive Knight"
            };

            foreach (string name in bossNames)
            {
                GameObject boss = GameObject.Find(name);
                if (boss != null && boss.activeInHierarchy)
                {
                    var hm = boss.GetComponent<HealthManager>();
                    if (hm != null && hm.hp > 0)
                    {
                        return boss;
                    }
                }
            }

            // Fallback: cerca qualsiasi HealthManager con hp > 50
            var healthManagers = GameObject.FindObjectsOfType<HealthManager>();
            foreach (var hm in healthManagers)
            {
                if (hm.hp > 50 && hm.gameObject.activeInHierarchy)
                {
                    string name = hm.gameObject.name.ToLower();

                    // Escludi il giocatore
                    if (name.Contains("knight") && name.Contains("hollow"))
                        continue;

                    if (name.Contains("boss") ||
                        name.Contains("hornet") ||
                        name.Contains("mantis") ||
                        name.Contains("mage") ||
                        name.Contains("ghost") ||
                        name.Contains("defender") ||
                        name.Contains("traitor"))
                    {
                        return hm.gameObject;
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Estrae informazioni sui pericoli: proiettili (boomerang mantidi), nemici, spike walls.
        /// </summary>
        private void ExtractHazards(HeroController hero, GameState state)
        {
            var playerPos = hero.transform.position;
            List<HazardInfo> hazards = new List<HazardInfo>();

            // ============ 1. BOOMERANG DELLE MANTIDI ============
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (var obj in allObjects)
            {
                if (obj == null || !obj.activeInHierarchy) continue;

                string name = obj.name.ToLower();

                // Rileva boomerang mantidi
                if (name.Contains("javelin") || name.Contains("boomerang") ||
                    name.Contains("scythe") || (name.Contains("mantis") && name.Contains("shot")))
                {
                    float distance = Vector2.Distance(playerPos, obj.transform.position);
                    if (distance <= HAZARD_DETECTION_RADIUS)
                    {
                        var hazard = new HazardInfo();
                        hazard.type = "boomerang";
                        hazard.relX = obj.transform.position.x - playerPos.x;
                        hazard.relY = obj.transform.position.y - playerPos.y;
                        hazard.distance = distance;

                        var rb = obj.GetComponent<Rigidbody2D>();
                        if (rb != null)
                        {
                            hazard.velocityX = rb.velocity.x;
                            hazard.velocityY = rb.velocity.y;
                        }

                        hazards.Add(hazard);
                        DesktopLogger.Log($"[Hazard] Boomerang detected: {obj.name} at ({hazard.relX:F1}, {hazard.relY:F1})");
                    }
                }
            }

            // ============ 2. PROJECTILES GENERICI (Backup) ============
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
                                var hazard = new HazardInfo();
                                hazard.type = "projectile";
                                hazard.relX = proj.transform.position.x - playerPos.x;
                                hazard.relY = proj.transform.position.y - playerPos.y;
                                hazard.distance = distance;

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

            // ============ 3. SPIKE WALLS E HAZARDS STATICI ============
            Vector2[] spikeDirections = new Vector2[]
            {
                Vector2.down,
                Vector2.up,
                Vector2.left,
                Vector2.right,
                new Vector2(-1, -1).normalized,
                new Vector2(1, -1).normalized,
                new Vector2(-1, 1).normalized,
                new Vector2(1, 1).normalized
            };

            foreach (var dir in spikeDirections)
            {
                RaycastHit2D hit = Physics2D.Raycast(playerPos, dir, HAZARD_DETECTION_RADIUS, hazardLayer);
                if (hit.collider != null)
                {
                    string hitName = hit.collider.gameObject.name.ToLower();
                    bool isSpike = hitName.Contains("spike") ||
                                   hitName.Contains("thorn") ||
                                   hitName.Contains("hazard") ||
                                   hit.collider.gameObject.layer == LayerMask.NameToLayer("Hazards");

                    if (isSpike && hit.distance < 10f)
                    {
                        var hazard = new HazardInfo();
                        hazard.type = "spikes";
                        hazard.relX = hit.point.x - playerPos.x;
                        hazard.relY = hit.point.y - playerPos.y;
                        hazard.distance = hit.distance;
                        hazard.velocityX = 0;
                        hazard.velocityY = 0;

                        hazards.Add(hazard);
                        DesktopLogger.Log($"[Hazard] Spikes detected at distance {hit.distance:F1} in direction ({dir.x:F1}, {dir.y:F1})");
                    }
                }
            }

            // ============ 4. ENEMIES (Mantis Lords stesse) ============
            var healthManagers = GameObject.FindObjectsOfType<HealthManager>();
            foreach (var hm in healthManagers)
            {
                if (hm.hp > 0 && hm.hp <= 50 && hm.gameObject.activeInHierarchy)
                {
                    float distance = Vector2.Distance(playerPos, hm.transform.position);
                    if (distance <= HAZARD_DETECTION_RADIUS)
                    {
                        var hazard = new HazardInfo();
                        hazard.type = "enemy";
                        hazard.relX = hm.transform.position.x - playerPos.x;
                        hazard.relY = hm.transform.position.y - playerPos.y;
                        hazard.distance = distance;

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
