using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Newtonsoft.Json;

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
        public float bossX;
        public float bossY;
        public float bossHealth;
        public float distanceToBoss;
        public float bossRelativeX;
        public float bossRelativeY;
        public bool isFacingBoss;
        public bool bossDefeated;
        public int mantisLordsKilled;
        public List<HazardInfo> nearbyHazards;

        public GameState()
        {
            terrainInfo = new float[5];
            nearbyHazards = new List<HazardInfo>();
            damageTaken = 0;
            lastHazardType = 1;
            mantisLordsKilled = 0;
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
        private const float HAZARD_DETECTION_RADIUS = 20f; // Aumentato un po' per vedere meglio
        private const int MAX_HAZARDS = 3;

        private LayerMask terrainLayer;
        private LayerMask hazardLayer;

        private int prevMantisLordsKilled = 0;

        public GameStateExtractor()
        {
            terrainLayer = LayerMask.GetMask("Terrain");
            hazardLayer = LayerMask.GetMask("Terrain", "Hazards");
        }

        public void ResetTracking()
        {
            prevMantisLordsKilled = 0;
            DesktopLogger.Log("[Extractor] Mantis Lords tracking reset");
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

            // 1. TERRA
            RaycastHit2D groundHit = Physics2D.Raycast(rayOrigin, Vector2.down, 5f, terrainLayer);
            state.terrainInfo[0] = (groundHit.collider != null) ? Mathf.Clamp01(groundHit.distance / 3f) : 1.0f;

            // 2. SOFFITTO
            RaycastHit2D ceilingHit = Physics2D.Raycast(rayOrigin, Vector2.up, 5f, terrainLayer);
            state.terrainInfo[1] = (ceilingHit.collider != null) ? Mathf.Clamp01(ceilingHit.distance / 3f) : 1.0f;

            Vector2 forward = hero.cState.facingRight ? Vector2.right : Vector2.left;
            Vector2 backward = hero.cState.facingRight ? Vector2.left : Vector2.right;

            // 3. MURO AVANTI
            RaycastHit2D wallHit = Physics2D.Raycast(rayOrigin, forward, 5f, terrainLayer);
            state.terrainInfo[2] = (wallHit.collider != null) ? Mathf.Clamp01(wallHit.distance / 3f) : 1.0f;

            // 4. MURO DIETRO
            RaycastHit2D wallBackHit = Physics2D.Raycast(rayOrigin, backward, 5f, terrainLayer);
            state.terrainInfo[3] = (wallBackHit.collider != null) ? Mathf.Clamp01(wallBackHit.distance / 3f) : 1.0f;

            // 5. DIAGONALE
            Vector2 diagDir = new Vector2(forward.x, -1f).normalized;
            RaycastHit2D diagHit = Physics2D.Raycast(rayOrigin, diagDir, 5f, terrainLayer);
            state.terrainInfo[4] = (diagHit.collider != null) ? Mathf.Clamp01(diagHit.distance / 3f) : 1.0f;
        }

        private void ExtractBossState(HeroController hero, GameState state)
        {
            // --- MODIFICA CRITICA: SMART BOSS FINDER ---
            GameObject boss = FindActiveMantisLord(hero.transform.position);
            // ------------------------------------------

            if (boss != null)
            {
                var bossPos = boss.transform.position;
                var playerPos = hero.transform.position;

                state.bossX = bossPos.x;
                state.bossY = bossPos.y;

                float dx = bossPos.x - playerPos.x;
                float dy = bossPos.y - playerPos.y;
                state.distanceToBoss = Mathf.Sqrt(dx * dx + dy * dy);

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
            }
            else
            {
                state.distanceToBoss = 100f;
                state.bossRelativeX = 0f;
                state.bossRelativeY = 0f;
            }

            int killedCount = CountDeadMantisLords();
            state.mantisLordsKilled = killedCount;
            if (killedCount != prevMantisLordsKilled)
            {
                prevMantisLordsKilled = killedCount;
            }
        }

        // =================================================================
        // NUOVA LOGICA: Trova il boss ATTIVO ignorando quelli sui troni
        // =================================================================
        private GameObject FindActiveMantisLord(Vector3 playerPos)
        {
            string[] lords = { "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2" };
            List<GameObject> activeCandidates = new List<GameObject>();

            // 1. Raccogli tutti i Mantis Lords vivi
            foreach (var name in lords)
            {
                var obj = GameObject.Find(name);
                if (obj != null)
                {
                    var hm = obj.GetComponent<HealthManager>();
                    if (hm != null && hm.hp > 0)
                    {
                        activeCandidates.Add(obj);
                    }
                }
            }

            if (activeCandidates.Count == 0) return null;
            if (activeCandidates.Count == 1) return activeCandidates[0];

            // 2. FILTRO ANTI-CAMPER:
            // I Mantis Lords sui troni sono generalmente più in ALTO (Y) o molto LONTANI (X) dal centro.
            // Strategia migliore: Il boss attivo è quello con la Y più BASSA (più vicino al pavimento).
            // Quelli sui troni sono rialzati.

            // Ordina per altezza (Y) crescente -> Il primo è quello più in basso (sul pavimento)
            activeCandidates.Sort((a, b) => a.transform.position.y.CompareTo(b.transform.position.y));

            // Restituisci quello più in basso.
            return activeCandidates[0];
        }

        private int CountDeadMantisLords()
        {
            string[] lords = { "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2" };
            int count = 0;
            foreach (var name in lords)
            {
                var obj = GameObject.Find(name);
                if (obj != null)
                {
                    var hm = obj.GetComponent<HealthManager>();
                    if (hm != null && hm.hp <= 0) count++;
                }
            }
            return count;
        }

        // Debug per hazard scan
        private float lastFullScanTime = 0f;
        private float lastHazardLogTime = 0f;

        private void ExtractHazards(HeroController hero, GameState state)
        {
            var playerPos = hero.transform.position;
            var hazards = new List<HazardInfo>();

            // ===============================================================
            // DEBUG: Scan completo della scena ogni 10 secondi
            // Questo ci mostra TUTTI gli oggetti che potrebbero essere spuntoni
            // ===============================================================
            if (Time.time - lastFullScanTime > 10.0f)
            {
                lastFullScanTime = Time.time;
                DesktopLogger.Log("========== FULL HAZARD SCAN ==========");

                // 1. Trova tutti gli oggetti con DamageHero
                var allDamageHero = GameObject.FindObjectsOfType<DamageHero>();
                DesktopLogger.Log($"[SCAN] Objects with DamageHero: {allDamageHero.Length}");
                foreach (var dmg in allDamageHero)
                {
                    var obj = dmg.gameObject;
                    float dist = Vector2.Distance(playerPos, obj.transform.position);
                    if (dist < 25f)
                    {
                        var rb = obj.GetComponent<Rigidbody2D>();
                        var col = obj.GetComponent<Collider2D>();
                        string rbInfo = rb != null ? (rb.isKinematic ? "STATIC" : "MOVING") : "NO_RB";
                        string colInfo = col != null ? (col.isTrigger ? "TRIGGER" : "SOLID") : "NO_COL";
                        DesktopLogger.Log($"  [{rbInfo}][{colInfo}] {obj.name} | Layer={LayerMask.LayerToName(obj.layer)} | Dist={dist:F1}");
                    }
                }

                // 2. Cerca specificamente nel layer "Interactive Object" o simili
                DesktopLogger.Log("[SCAN] Checking all layers for damage zones...");
                var allColliders = GameObject.FindObjectsOfType<Collider2D>();
                foreach (var col in allColliders)
                {
                    string layerName = LayerMask.LayerToName(col.gameObject.layer);
                    string objName = col.gameObject.name.ToLower();

                    // Cerca oggetti che potrebbero essere spuntoni
                    if (layerName.Contains("Hazard") || layerName.Contains("Damage") ||
                        objName.Contains("spike") || objName.Contains("thorn") ||
                        objName.Contains("hurt") || objName.Contains("damage") ||
                        objName.Contains("kill") || objName.Contains("death"))
                    {
                        float dist = Vector2.Distance(playerPos, col.transform.position);
                        if (dist < 20f)
                        {
                            DesktopLogger.Log($"  [POTENTIAL SPIKE] {col.gameObject.name} | Layer={layerName} | Trigger={col.isTrigger} | Dist={dist:F1}");
                        }
                    }
                }
                DesktopLogger.Log("========== END SCAN ==========");
            }

            // ===============================================================
            // Rilevamento hazard normale
            // ===============================================================
            // Usa OverlapCircleAll CON tutti i layer (non filtrato)
            Collider2D[] hits = Physics2D.OverlapCircleAll(playerPos, HAZARD_DETECTION_RADIUS);

            bool shouldLog = (Time.time - lastHazardLogTime > 2.0f);

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

                // ============================================
                // RILEVAMENTO SPUNTONI
                // ============================================
                // Metodo 1: Ha DamageHero ed è statico
                if (damageHero != null && isStatic)
                {
                    isSpike = true;
                    hazardType = "spike";
                }

                // Metodo 2: Nome contiene parole chiave spuntoni
                if (name.Contains("spike") || name.Contains("thorn") || name.Contains("spikes") ||
                    name.Contains("hurt") || name.Contains("dvl_dmg"))
                {
                    isSpike = true;
                    hazardType = "spike";
                }

                // Metodo 3: Layer è "Hazard" o simile
                string layerName = LayerMask.LayerToName(hit.gameObject.layer);
                if (layerName.Contains("Hazard") || layerName.Contains("Damage"))
                {
                    isSpike = true;
                    hazardType = "spike";
                }

                // ============================================
                // RILEVAMENTO PROIETTILI (si muovono!)
                // ============================================
                if (name.Contains("shot") || name.Contains("scythe") ||
                    name.Contains("boomerang") || name.Contains("projectile") ||
                    name.Contains("slash") || name.Contains("disc"))
                {
                    isProjectile = true;
                    hazardType = "projectile";
                }
                // Ha DamageHero ma si muove = proiettile
                else if (damageHero != null && !isStatic)
                {
                    isProjectile = true;
                    hazardType = "projectile";
                }

                // ============================================
                // RILEVAMENTO NEMICI
                // ============================================
                if (hit.gameObject.layer == LayerMask.NameToLayer("Enemies"))
                {
                    // Ignora Mantis Lords sui troni
                    if (name.Contains("mantis") && hit.transform.position.y > playerPos.y + 5.0f)
                    {
                        continue;
                    }
                    if (!isSpike && !isProjectile)
                    {
                        isEnemy = true;
                        hazardType = "enemy";
                    }
                }

                bool isThreat = isSpike || isProjectile || isEnemy;

                if (isThreat)
                {
                    float dist = Vector2.Distance(playerPos, hit.transform.position);

                    var hInfo = new HazardInfo
                    {
                        type = hazardType,
                        relX = hit.transform.position.x - playerPos.x,
                        relY = hit.transform.position.y - playerPos.y,
                        distance = dist
                    };

                    if (rb != null)
                    {
                        hInfo.velocityX = rb.velocity.x;
                        hInfo.velocityY = rb.velocity.y;
                    }

                    hazards.Add(hInfo);

                    // Log hazard vicini
                    if (shouldLog && dist < 5.0f)
                    {
                        DesktopLogger.Log($"[HAZARD] {hit.name} | Type={hazardType} | Dist={dist:F2} | Static={isStatic}");
                    }
                }
            }

            if (shouldLog && hazards.Count > 0)
            {
                lastHazardLogTime = Time.time;
                int spikes = hazards.Count(h => h.type == "spike");
                int projectiles = hazards.Count(h => h.type == "projectile");
                int enemies = hazards.Count(h => h.type == "enemy");
                DesktopLogger.Log($"[HAZARD TOTAL] Spikes={spikes} Projectiles={projectiles} Enemies={enemies}");
            }

            state.nearbyHazards = hazards.OrderBy(h => h.distance).Take(MAX_HAZARDS * 2).ToList();
        }

        private GameState GetDefaultState()
        {
            return new GameState
            {
                distanceToBoss = 100f,
                terrainInfo = new float[] { 1, 1, 1, 1, 1 }
            };
        }
    }
}