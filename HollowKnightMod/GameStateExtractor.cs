using System;
using System.Collections.Generic;
using UnityEngine;

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
        public int playerMaxHealth;
        public int playerSoul;
        public bool canDash;
        public bool canAttack;
        public bool isGrounded;
        public bool hasDoubleJump;
        public float bossX;
        public float bossY;
        public int bossHealth;
        public int bossMaxHealth;
        public string bossState;
        public float distanceToBoss;
        public bool isDead;
        public bool bossDefeated;
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
                    // FIX: Corrected Rigidbody2D validation - use .simulated instead of .isActiveAndEnabled
                    if (rb != null && rb.simulated && rb.gameObject.activeInHierarchy)
                    {
                        velocityX = rb.velocity.x;
                        velocityY = rb.velocity.y;
                    }
                }
                catch (Exception rbEx)
                {
                    // FIX: Silently handle Rigidbody access errors during damage frames
                    Modding.Logger.LogWarn($"[SyntheticSoul] Rigidbody2D access failed (likely during damage): {rbEx.Message}");
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

                    // FIX: Safely check hero state during recoil/damage
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
                    timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
                };

                // Extract boss data if available
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

                    // Calculate distance
                    float dx = state.bossX - state.playerX;
                    float dy = state.bossY - state.playerY;
                    state.distanceToBoss = Mathf.Sqrt(dx * dx + dy * dy);

                    // Try to get boss state from PlayMaker FSM
                    var fsm = PlayMakerUtils.FindFsmOnGameObject(bossObject, "Control");
                    if (fsm == null)
                        fsm = PlayMakerUtils.FindFsmOnGameObject(bossObject, "Battle Control");
                    
                    if (fsm != null)
                    {
                        state.bossState = fsm.ActiveStateName ?? "UNKNOWN";
                    }
                }

                return state;
            }
            catch (Exception e)
            {
                // FIX: Always return valid state instead of crashing
                Modding.Logger.LogError($"[SyntheticSoul] Error extracting state: {e.Message}\n{e.StackTrace}");
                DesktopLogger.LogError($"ExtractState crash: {e.Message}");
                return CreateDeadState();
            }
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
                            Modding.Logger.Log($"Found boss: {hm.gameObject.name} with HP: {hm.hp}");
                            DesktopLogger.Log($"Boss found: {hm.gameObject.name} HP:{hm.hp}");
                            break;
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Modding.Logger.LogError($"Error finding boss: {e.Message}");
                DesktopLogger.LogError($"FindBoss error: {e.Message}");
            }
        }
    }
}