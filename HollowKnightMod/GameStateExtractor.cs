using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace SyntheticSoulMod
{
    // Struct per informazioni sui pericoli
    [Serializable]
    public class HazardInfo
    {
        public float relX;      // Posizione X relativa al player
        public float relY;      // Posizione Y relativa al player
        public int type;        // 1 = Nemico, 2 = Proiettile
    }

    [Serializable]
    public class GameState
    {
        // ========== PLAYER DATA ==========
        public float playerX;
        public float playerY;
        public float playerVelocityX;
        public float playerVelocityY;
        public int playerHealth;
        public int playerMaxHealth;
        public int playerSoul;
        public bool canDash;
        public bool canAttack;
        public bool isGrounded;
        public bool hasDoubleJump;

        // ========== BOSS DATA ==========
        public float bossX;
        public float bossY;
        public int bossHealth;
        public int bossMaxHealth;
        public string bossState;
        public float distanceToBoss;
        public bool isDead;
        public bool bossDefeated;

        // ========== NUOVO: PERCEZIONE AMBIENTALE ==========
        // Array di 5 float: distanze dal terreno in 5 direzioni
        // [0] = Sotto, [1] = Avanti-Basso, [2] = Avanti, [3] = Avanti-Alto, [4] = Sopra
        public float[] terrainInfo;

        // Lista dei 5 pericoli più vicini (nemici minori e proiettili)
        public List<HazardInfo> nearbyHazards;

        public long timestamp;
    }

    public class GameStateExtractor
    {
        private GameObject bossObject;
        private HealthManager bossHealthManager;
        private List<string> bossNames = new List<string>
        {
            "False Knight",
            "Hornet",
            "Mawlek",
            "Mantis Lord",
            "Soul Master",
            "Broken Vessel",
            "Dung Defender",
            "Watcher",
            "Collector",
            "Traitor Lord"
        };

        // ========== COSTANTI PER PERCEZIONE AMBIENTALE ==========
        private const float TERRAIN_RAYCAST_MAX_DISTANCE = 5.0f;
        private const float HAZARD_DETECTION_RADIUS = 10.0f;
        private const int MAX_HAZARDS_TO_TRACK = 5;
        private const int TERRAIN_LAYER = 8;        // Layer "Terrain" in Hollow Knight
        private const int ENEMIES_LAYER = 11;       // Layer "Enemies"
        private const int PROJECTILES_LAYER = 17;   // Layer "Projectiles"

        private LayerMask terrainMask;
        private LayerMask hazardMask;

        public GameStateExtractor()
        {
            InitializeLayerMasks();
        }

        private void InitializeLayerMasks()
        {
            try
            {
                // LayerMask per terreno
                terrainMask = LayerMask.GetMask("Terrain");
                if (terrainMask == 0)
                {
                    terrainMask = 1 << TERRAIN_LAYER;
                }

                // LayerMask per pericoli (nemici + proiettili)
                hazardMask = 0;
                try { hazardMask |= LayerMask.GetMask("Enemies"); } catch { }
                try { hazardMask |= LayerMask.GetMask("Enemy"); } catch { }
                try { hazardMask |= LayerMask.GetMask("Projectiles"); } catch { }
                try { hazardMask |= LayerMask.GetMask("Attack"); } catch { }

                if (hazardMask == 0)
                {
                    // Fallback ai layer numerici
                    hazardMask = (1 << ENEMIES_LAYER) | (1 << PROJECTILES_LAYER);
                }
            }
            catch (Exception ex)
            {
                Modding.Logger.LogWarn($"[SyntheticSoul] Error initializing layer masks: {ex.Message}");
                terrainMask = 1 << TERRAIN_LAYER;
                hazardMask = (1 << ENEMIES_LAYER) | (1 << PROJECTILES_LAYER);
            }
        }

        private GameState CreateDeadState()
        {
            return new GameState
            {
                playerX = 0f,
                playerY = 0f,
                playerVelocityX = 0f,
                playerVelocityY = 0f,
                playerHealth = 0,
                playerMaxHealth = 5,
                playerSoul = 0,
                canDash = false,
                canAttack = false,
                isGrounded = false,
                hasDoubleJump = false,
                bossX = 0f,
                bossY = 0f,
                bossHealth = 0,
                bossMaxHealth = 1,
                bossState = "UNKNOWN",
                distanceToBoss = 999f,
                isDead = true,
                bossDefeated = false,
                terrainInfo = new float[5] { 5.0f, 5.0f, 5.0f, 5.0f, 5.0f },
                nearbyHazards = new List<HazardInfo>(),
                timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
            };
        }

        public GameState ExtractState()
        {
            try
            {
                var hero = HeroController.instance;
                var playerData = PlayerData.instance;

                if (hero == null || playerData == null)
                {
                    Modding.Logger.Log("[SyntheticSoul] Hero or PlayerData is null - returning dead state");
                    return CreateDeadState();
                }

                if (bossObject == null || bossHealthManager == null || bossHealthManager.hp <= 0)
                {
                    FindBoss();
                }

                // FIX: Wrap Rigidbody2D access in try-catch to prevent crash during damage/recoil
                float velocityX = 0f;
                float velocityY = 0f;
                try
                {
                    var rb = hero.GetComponent<Rigidbody2D>();
                    if (rb != null && rb.simulated && rb.gameObject.activeInHierarchy)
                    {
                        velocityX = rb.velocity.x;
                        velocityY = rb.velocity.y;
                    }
                }
                catch (Exception rbEx)
                {
                    Modding.Logger.LogWarn($"[SyntheticSoul] Rigidbody2D access failed: {rbEx.Message}");
                }

                var state = new GameState
                {
                    // Player data
                    playerX = hero.transform.position.x,
                    playerY = hero.transform.position.y,
                    playerVelocityX = velocityX,
                    playerVelocityY = velocityY,
                    playerHealth = playerData.health,
                    playerMaxHealth = playerData.maxHealth,
                    playerSoul = playerData.MPCharge,

                    canDash = !hero.cState.dashing && !hero.cState.shadowDashing && !hero.cState.recoiling,
                    canAttack = !hero.cState.attacking && !hero.cState.recoiling && !hero.cState.dead,
                    isGrounded = hero.cState.onGround,
                    hasDoubleJump = playerData.hasDoubleJump,

                    // Boss data defaults
                    bossX = 0f,
                    bossY = 0f,
                    bossHealth = 0,
                    bossMaxHealth = 1,
                    bossState = "UNKNOWN",
                    distanceToBoss = 999f,
                    isDead = playerData.health <= 0,
                    bossDefeated = false,

                    // Initialize perception arrays
                    terrainInfo = new float[5],
                    nearbyHazards = new List<HazardInfo>(),

                    timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
                };

                // Extract boss data
                if (bossObject != null && bossHealthManager != null)
                {
                    state.bossX = bossObject.transform.position.x;
                    state.bossY = bossObject.transform.position.y;
                    state.bossHealth = bossHealthManager.hp;

                    try
                    {
                        var maxHpField = typeof(HealthManager).GetField("enemyHealthMax",
                            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        if (maxHpField != null)
                        {
                            state.bossMaxHealth = (int)maxHpField.GetValue(bossHealthManager);
                        }
                    }
                    catch
                    {
                        state.bossMaxHealth = bossHealthManager.hp > 0 ? bossHealthManager.hp : 100;
                    }

                    state.bossDefeated = bossHealthManager.hp <= 0;

                    float dx = state.bossX - state.playerX;
                    float dy = state.bossY - state.playerY;
                    state.distanceToBoss = Mathf.Sqrt(dx * dx + dy * dy);

                    var fsm = PlayMakerUtils.FindFsmOnGameObject(bossObject, "Control");
                    if (fsm == null)
                        fsm = PlayMakerUtils.FindFsmOnGameObject(bossObject, "Battle Control");
                    if (fsm != null)
                    {
                        state.bossState = fsm.ActiveStateName ?? "UNKNOWN";
                    }
                }

                // ========== NUOVO: ESTRAI PERCEZIONE AMBIENTALE ==========
                ExtractTerrainInfo(hero, ref state);
                ExtractNearbyHazards(hero, ref state);

                return state;
            }
            catch (Exception e)
            {
                Modding.Logger.LogError($"[SyntheticSoul] Error extracting state: {e.Message}\n{e.StackTrace}");
                DesktopLogger.LogError($"ExtractState crash: {e.Message}");
                return CreateDeadState();
            }
        }

        // ========== NUOVO: RILEVAMENTO TERRENO CON RAYCAST ==========
        private void ExtractTerrainInfo(HeroController hero, ref GameState state)
        {
            try
            {
                Vector2 playerPos = hero.transform.position;
                bool facingRight = hero.cState.facingRight;

                // Direzione orizzontale basata su dove guarda il player
                Vector2 forwardDir = facingRight ? Vector2.right : Vector2.left;

                // 5 direzioni di raycast
                Vector2[] directions = new Vector2[5]
                {
                    Vector2.down,                                      // [0] Sotto
                    (forwardDir + Vector2.down).normalized,            // [1] Avanti-Basso (45° diagonale)
                    forwardDir,                                        // [2] Avanti (orizzontale)
                    (forwardDir + Vector2.up).normalized,              // [3] Avanti-Alto (45° diagonale)
                    Vector2.up                                         // [4] Sopra
                };

                // Lancia raycast per ogni direzione
                for (int i = 0; i < 5; i++)
                {
                    RaycastHit2D hit = Physics2D.Raycast(
                        playerPos,
                        directions[i],
                        TERRAIN_RAYCAST_MAX_DISTANCE,
                        terrainMask
                    );

                    if (hit.collider != null)
                    {
                        // Normalizza la distanza (0-1, dove 1 = max distance)
                        state.terrainInfo[i] = Mathf.Clamp01(hit.distance / TERRAIN_RAYCAST_MAX_DISTANCE);
                        
                        // Debug log per i primi frame
                        if (UnityEngine.Random.value < 0.01f) // Log 1% del tempo
                        {
                            Modding.Logger.Log($"[Terrain] Dir {i}: hit at {hit.distance:F2} units");
                        }
                    }
                    else
                    {
                        // Nessun terreno trovato = distanza massima
                        state.terrainInfo[i] = 1.0f;
                    }
                }
            }
            catch (Exception ex)
            {
                Modding.Logger.LogWarn($"[SyntheticSoul] Error in ExtractTerrainInfo: {ex.Message}");
                // Fallback: distanze massime
                for (int i = 0; i < 5; i++)
                {
                    state.terrainInfo[i] = 1.0f;
                }
            }
        }

        // ========== NUOVO: RILEVAMENTO PERICOLI CON OVERLAPCIRCLE ==========
        private void ExtractNearbyHazards(HeroController hero, ref GameState state)
        {
            try
            {
                Vector2 playerPos = hero.transform.position;
                List<HazardInfo> hazards = new List<HazardInfo>();

                // Trova tutti i collider nel raggio
                Collider2D[] colliders = Physics2D.OverlapCircleAll(
                    playerPos,
                    HAZARD_DETECTION_RADIUS,
                    hazardMask
                );

                foreach (var col in colliders)
                {
                    if (col == null || col.gameObject == null)
                        continue;

                    // FILTRO 1: Ignora il player stesso
                    if (col.gameObject.name.Contains("Knight") || col.gameObject.layer == LayerMask.NameToLayer("Player"))
                        continue;

                    // FILTRO 2: Ignora il boss principale (già tracciato separatamente)
                    if (bossObject != null && (col.gameObject == bossObject || col.transform.IsChildOf(bossObject.transform)))
                        continue;

                    // FILTRO 3: Ignora oggetti inattivi o invisibili
                    if (!col.gameObject.activeInHierarchy)
                        continue;

                    // Calcola posizione relativa
                    float relX = col.transform.position.x - playerPos.x;
                    float relY = col.transform.position.y - playerPos.y;
                    float distance = Mathf.Sqrt(relX * relX + relY * relY);

                    // Determina il tipo (1 = Nemico, 2 = Proiettile)
                    int hazardType = DetermineHazardType(col.gameObject);

                    hazards.Add(new HazardInfo
                    {
                        relX = relX,
                        relY = relY,
                        type = hazardType
                    });
                }

                // Ordina per distanza e prendi i 5 più vicini
                state.nearbyHazards = hazards
                    .OrderBy(h => h.relX * h.relX + h.relY * h.relY) // Ordina per distanza al quadrato (più veloce)
                    .Take(MAX_HAZARDS_TO_TRACK)
                    .ToList();

                // Debug log
                if (state.nearbyHazards.Count > 0 && UnityEngine.Random.value < 0.05f)
                {
                    Modding.Logger.Log($"[Hazards] Detected {state.nearbyHazards.Count} threats: " +
                        $"Closest at ({state.nearbyHazards[0].relX:F1}, {state.nearbyHazards[0].relY:F1})");
                }
            }
            catch (Exception ex)
            {
                Modding.Logger.LogWarn($"[SyntheticSoul] Error in ExtractNearbyHazards: {ex.Message}");
                state.nearbyHazards = new List<HazardInfo>();
            }
        }

        // Determina se un GameObject è un nemico (1) o un proiettile (2)
        private int DetermineHazardType(GameObject obj)
        {
            string name = obj.name.ToLower();
            int layer = obj.layer;

            // ========== PROIETTILI (type = 2) ==========
            // Controlla layer proiettili
            if (layer == PROJECTILES_LAYER)
                return 2;

            // Controlla nomi comuni di proiettili in Hollow Knight
            if (name.Contains("shot") || name.Contains("bullet") || name.Contains("projectile") ||
                name.Contains("orb") || name.Contains("spell") || name.Contains("acid") ||
                name.Contains("fireball") || name.Contains("spit") || name.Contains("blob") ||
                name.Contains("spike ball") || name.Contains("nail"))
                return 2;

            // ========== NEMICI (type = 1) ==========
            // Controlla se ha HealthManager (tipico dei nemici)
            if (obj.GetComponent<HealthManager>() != null)
                return 1;

            // Controlla layer nemici
            if (layer == ENEMIES_LAYER)
                return 1;

            // Controlla nomi comuni di nemici
            if (name.Contains("enemy") || name.Contains("crawler") || name.Contains("fly") ||
                name.Contains("buzzer") || name.Contains("aspid") || name.Contains("hatcher") ||
                name.Contains("hopper") || name.Contains("mosquito"))
                return 1;

            // Default: considera come proiettile (più conservativo)
            return 2;
        }

        private void FindBoss()
        {
            try
            {
                var allHealthManagers = GameObject.FindObjectsOfType<HealthManager>();
                foreach (var hm in allHealthManagers)
                {
                    if (hm.hp > 100 && !hm.IsInvincible)
                    {
                        bool isBoss = false;
                        foreach (var bossName in bossNames)
                        {
                            if (hm.gameObject.name.Contains(bossName))
                            {
                                isBoss = true;
                                break;
                            }
                        }

                        if (isBoss || hm.hp > 200)
                        {
                            bossObject = hm.gameObject;
                            bossHealthManager = hm;
                            Modding.Logger.Log($"[SyntheticSoul] Found boss: {hm.gameObject.name} with HP: {hm.hp}");
                            DesktopLogger.Log($"Boss found: {hm.gameObject.name} HP:{hm.hp}");
                            break;
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.LogError($"[SyntheticSoul] Error finding boss: {e.Message}");
                DesktopLogger.LogError($"FindBoss error: {e.Message}");
            }
        }
    }
}
