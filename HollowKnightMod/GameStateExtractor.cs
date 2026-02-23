using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Newtonsoft.Json;
using HutongGames.PlayMaker;

namespace SyntheticSoulMod
{
    [Serializable]
    public class GameState
    {
        // ═══════ PLAYER ═══════
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

        // ═══════ BOSS (target attivo — THREAT LOCKED) ═══════
        // Questi campi puntano SEMPRE alla mantide più pericolosa,
        // non alla più vicina o alla più bassa. In fase 2 se una
        // mantide ti dasha addosso, questi campi trackano LEI.
        public float bossX;
        public float bossY;
        public float bossVelocityX;
        public float bossVelocityY;
        public float bossHealth;
        public float distanceToBoss;
        public float bossRelativeX;
        public float bossRelativeY;
        public bool isFacingBoss;
        public bool bossDefeated;
        public int mantisLordsKilled;

        // Boss Intent legacy — con PRIORITÀ: Drop(2) > Dash(1) > Throw(3) > Idle(0)
        // Scansiona TUTTE le mantidi e riporta l'attacco più pericoloso in corso.
        public int bossAction;

        // ═══════ HAZARDS ═══════
        public List<HazardInfo> nearbyHazards;

        // ═══════ ATTACK PATTERN DETECTION ═══════
        // Pattern mantide primaria (la PIÙ PERICOLOSA, stessa della threat lock)
        // 0=IDLE, 1=DASH_HORIZONTAL, 2=DASH_DIAGONAL, 3=DAGGER_THROW,
        // 4=WALL_ATTACK, 5=PLUNGE_ATTACK, 6=WIND_UP, 7=RECOVERING
        public int primaryMantisPattern;
        public float primaryMantisVelX;
        public float primaryMantisVelY;
        public float primaryMantisAttackDuration;
        public bool primaryMantisWindUp;
        public bool primaryMantisActive;
        public bool primaryMantisRecovering;

        // Pattern mantide secondaria (fase 2, la seconda per pericolosità)
        public int secondaryMantisPattern;
        public float secondaryMantisVelX;
        public float secondaryMantisVelY;
        public float secondaryMantisRelX;
        public float secondaryMantisRelY;
        public bool secondaryMantisActive;

        // Info globali combattimento
        public int activeMantisCount;
        public bool anyMantisAttacking;

        // ═══════ VICTORY TRACKER ═══════
        public int sessionWins;
        public int currentStreak;
        public float winRate;

        public GameState()
        {
            terrainInfo = new float[5];
            nearbyHazards = new List<HazardInfo>();
            damageTaken = 0;
            lastHazardType = 1;
            mantisLordsKilled = 0;
            bossAction = 0;
            primaryMantisPattern = 0;
            secondaryMantisPattern = 0;
            activeMantisCount = 0;
            anyMantisAttacking = false;
            sessionWins = 0;
            currentStreak = 0;
            winRate = 0f;
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
        private const float HAZARD_DETECTION_RADIUS = 25f;
        private const int MAX_HAZARDS = 3;

        private LayerMask terrainLayer;
        private LayerMask hazardLayer;

        private int prevMantisLordsKilled = 0;
        private float lastKnownBossHealth = 400f;

        // Kill tracking fase 2
        private bool phase2Detected = false;
        private float lastKnownS1Hp = -1f;
        private float lastKnownS2Hp = -1f;
        private int confirmedPhase2Kills = 0;

        // Attack Detector
        private MantisAttackDetector attackDetector;

        // Risultato dell'ultimo ExtractAttackPatterns, usato da ExtractBossState
        // per fare threat lock coerente tra campi legacy e V2
        private MantisAttackDetector.MantisAttackInfo lastPrimaryThreat = null;

        public GameStateExtractor()
        {
            terrainLayer = LayerMask.GetMask("Terrain");
            hazardLayer = LayerMask.GetMask("Terrain", "Hazards");
            attackDetector = new MantisAttackDetector();
        }

        public void ResetTracking()
        {
            prevMantisLordsKilled = 0;
            lastKnownBossHealth = 400f;
            phase2Detected = false;
            lastKnownS1Hp = -1f;
            lastKnownS2Hp = -1f;
            confirmedPhase2Kills = 0;
            lastPrimaryThreat = null;
            attackDetector?.Reset();
            DesktopLogger.Log("[Extractor] Tracking reset");
        }

        public GameState ExtractState()
        {
            var state = new GameState();
            var hero = HeroController.instance;
            if (hero == null) return GetDefaultState();

            ExtractPlayerState(hero, state);
            ExtractTerrainInfo(hero, state);

            // ORDINE CRITICO: Pattern PRIMA di BossState.
            // ExtractAttackPatterns popola lastPrimaryThreat,
            // ExtractBossState lo usa per il threat lock.
            ExtractAttackPatterns(hero, state);
            ExtractBossState(hero, state);

            ExtractHazards(hero, state);

            return state;
        }

        // ═══════════════════════════════════════════════════════
        // PLAYER
        // ═══════════════════════════════════════════════════════
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

            state.canDash = !hero.cState.dashing && !hero.cState.backDashing &&
                            !hero.cState.shadowDashing && !hero.cState.dashCooldown;
            state.canAttack = !hero.cState.attacking && !hero.cState.recoiling && !hero.cState.dead;
            state.isGrounded = hero.cState.onGround;
            state.isDead = hero.cState.dead;
            state.facingRight = hero.cState.facingRight;
        }

        // ═══════════════════════════════════════════════════════
        // TERRAIN
        // ═══════════════════════════════════════════════════════
        private void ExtractTerrainInfo(HeroController hero, GameState state)
        {
            Vector2 playerPos = hero.transform.position;
            Vector2 rayOrigin = new Vector2(playerPos.x, playerPos.y + 0.5f);

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

        // ═══════════════════════════════════════════════════════
        // BOSS STATE — THREAT LOCKED
        //
        // Logica (ripristinata dal vecchio codice + integrata col detector):
        //
        // 1. Calcola vita/kills come prima
        // 2. Se il detector ha trovato una primary threat (chi sta attaccando),
        //    LOCKA bossRelativeX/Y/distanceToBoss SU DI LEI
        // 3. Se nessuna sta attaccando, usa FindActiveMantisLord (gerarchia fase1→2)
        // 4. bossAction = intent più pericoloso tra TUTTE le mantidi (PrioritizeIntent)
        //
        // RISULTATO: i campi legacy (features 12-19 nel preprocessor) e i campi V2
        // (features 34-50) puntano alla STESSA mantide = informazione coerente.
        // ═══════════════════════════════════════════════════════
        private void ExtractBossState(HeroController hero, GameState state)
        {
            // ─── 1. VITA E KILLS ───
            float totalHealth = CalculateTotalMantisHealth();

            if (totalHealth > 0)
            {
                state.bossHealth = totalHealth;
                state.bossDefeated = false;
                lastKnownBossHealth = totalHealth;
            }
            else
            {
                if (CountDeadMantisLords() >= 3)
                {
                    state.bossHealth = 0;
                    state.bossDefeated = true;
                }
                else
                {
                    state.bossHealth = lastKnownBossHealth;
                }
            }

            state.mantisLordsKilled = CountDeadMantisLords();

            // ─── 2. THREAT SCAN: bossAction = intent più pericoloso globale ───
            // Scansiona TUTTE le mantidi attive e prende l'intent con priorità
            // più alta: Drop(2) > Dash(1) > Throw(3) > Idle(0)
            int highestPriorityIntent = 0;
            string[] lords = { "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2" };

            foreach (var name in lords)
            {
                var obj = GameObject.Find(name);
                if (obj != null && obj.activeSelf)
                {
                    var hm = obj.GetComponent<HealthManager>();
                    if (hm != null && hm.hp > 0)
                    {
                        int intent = GetBossIntent(obj);
                        if (intent != 0)
                        {
                            highestPriorityIntent = PrioritizeIntent(highestPriorityIntent, intent);
                        }
                    }
                }
            }
            state.bossAction = highestPriorityIntent;

            // ─── 3. TARGET SELECTION — THREAT LOCK ───
            // Priorità: detector primary threat > mantide che sta attaccando > fallback gerarchico
            GameObject activeBoss = null;
            Vector3 playerPos = hero.transform.position;

            // 3a. Se il detector ha trovato una mantide pericolosa (sta attaccando o wind-up),
            //     usa LEI come target. Così campi legacy e V2 sono coerenti.
            if (lastPrimaryThreat != null && lastPrimaryThreat.isAlive &&
                (lastPrimaryThreat.isActiveAttack || lastPrimaryThreat.isWindingUp))
            {
                var threatObj = GameObject.Find(lastPrimaryThreat.name);
                if (threatObj != null && threatObj.activeSelf)
                {
                    activeBoss = threatObj;
                }
            }

            // 3b. Fallback: usa la gerarchia fase1→fase2
            if (activeBoss == null)
            {
                activeBoss = FindActiveMantisLord(playerPos);
            }

            // ─── 4. POPOLA CAMPI BOSS ───
            if (activeBoss != null)
            {
                var bossPos = activeBoss.transform.position;

                state.bossX = bossPos.x;
                state.bossY = bossPos.y;

                float dx = bossPos.x - playerPos.x;
                float dy = bossPos.y - playerPos.y;
                state.distanceToBoss = Mathf.Sqrt(dx * dx + dy * dy);
                state.bossRelativeX = Mathf.Clamp(dx / 40.0f, -1.0f, 1.0f);
                state.bossRelativeY = Mathf.Clamp(dy / 30.0f, -1.0f, 1.0f);
                state.isFacingBoss = (state.bossRelativeX > 0 && state.facingRight) ||
                                     (state.bossRelativeX < 0 && !state.facingRight);

                // Velocità del boss target
                var rb = activeBoss.GetComponent<Rigidbody2D>();
                if (rb != null)
                {
                    state.bossVelocityX = rb.velocity.x;
                    state.bossVelocityY = rb.velocity.y;
                }

                // Se il threat scan non ha trovato niente, usa l'intent di questa specifica mantide
                if (state.bossAction == 0)
                {
                    state.bossAction = GetBossIntent(activeBoss);
                }
            }
            else
            {
                state.distanceToBoss = 100f;
                state.bossAction = 0;
                state.bossX = 0;
                state.bossY = 0;
            }
        }

        // ═══════════════════════════════════════════════════════
        // ATTACK PATTERN DETECTION
        // Popola i campi primaryMantis* e secondaryMantis*
        // E salva lastPrimaryThreat per il threat lock in ExtractBossState.
        // ═══════════════════════════════════════════════════════
        private void ExtractAttackPatterns(HeroController hero, GameState state)
        {
            lastPrimaryThreat = null;  // Reset ogni frame

            if (attackDetector == null) return;

            Vector3 playerPos = hero.transform.position;
            var allPatterns = attackDetector.DetectAllPatterns(playerPos);

            int activeCount = 0;
            bool anyAttacking = false;

            MantisAttackDetector.MantisAttackInfo primary = null;
            MantisAttackDetector.MantisAttackInfo secondary = null;

            // Ordina per pericolosità: active attack > wind-up > qualsiasi pattern > idle
            // A parità, la più vicina vince
            allPatterns.Sort((a, b) =>
            {
                if (!a.isAlive && b.isAlive) return 1;
                if (a.isAlive && !b.isAlive) return -1;
                int scoreA = a.isActiveAttack ? 3 : (a.isWindingUp ? 2 : (a.attackPattern > 0 ? 1 : 0));
                int scoreB = b.isActiveAttack ? 3 : (b.isWindingUp ? 2 : (b.attackPattern > 0 ? 1 : 0));
                if (scoreA != scoreB) return scoreB.CompareTo(scoreA);
                return a.distanceToPlayer.CompareTo(b.distanceToPlayer);
            });

            foreach (var info in allPatterns)
            {
                if (!info.isAlive) continue;
                activeCount++;
                if (info.isActiveAttack || info.isWindingUp) anyAttacking = true;

                if (primary == null) primary = info;
                else if (secondary == null) secondary = info;
            }

            state.activeMantisCount = activeCount;
            state.anyMantisAttacking = anyAttacking;

            if (primary != null)
            {
                state.primaryMantisPattern = primary.attackPattern;
                state.primaryMantisVelX = primary.velocityX;
                state.primaryMantisVelY = primary.velocityY;
                state.primaryMantisAttackDuration = primary.attackDuration;
                state.primaryMantisWindUp = primary.isWindingUp;
                state.primaryMantisActive = primary.isActiveAttack;
                state.primaryMantisRecovering = primary.isRecovering;

                // Salva per il threat lock
                lastPrimaryThreat = primary;
            }

            if (secondary != null)
            {
                state.secondaryMantisPattern = secondary.attackPattern;
                state.secondaryMantisVelX = secondary.velocityX;
                state.secondaryMantisVelY = secondary.velocityY;
                state.secondaryMantisRelX = secondary.relativeX;
                state.secondaryMantisRelY = secondary.relativeY;
                state.secondaryMantisActive = secondary.isActiveAttack;
            }
        }

        // ═══════════════════════════════════════════════════════
        // INTENT + PRIORITÀ (RIPRISTINATO DAL VECCHIO CODICE)
        // ═══════════════════════════════════════════════════════

        private int GetBossIntent(GameObject boss)
        {
            var fsm = boss.GetComponent<PlayMakerFSM>();
            if (fsm == null) return 0;

            string activeState = fsm.ActiveStateName.ToLower();

            if (activeState.Contains("dash") || activeState.Contains("lunge")) return 1;
            if (activeState.Contains("land") || activeState.Contains("drop") || activeState.Contains("plunge")) return 2;
            if (activeState.Contains("throw") || activeState.Contains("wall")) return 3;

            return 0;
        }

        /// <summary>
        /// Gerarchia di pericolosità degli attacchi:
        ///   Drop/Plunge (2) > Dash/Lunge (1) > Throw (3) > Idle (0)
        ///
        /// Drop è il più pericoloso: veloce, verticale, poco tempo per reagire.
        /// Dash è secondo: lineare ma rapido, devi saltare o dashare.
        /// Throw è il meno urgente: il boomerang è lento, hai tempo per posizionarti.
        /// </summary>
        private int PrioritizeIntent(int currentBest, int newIntent)
        {
            if (currentBest == 2) return 2;
            if (newIntent == 2) return 2;
            if (currentBest == 1) return 1;
            if (newIntent == 1) return 1;
            if (currentBest == 3) return 3;
            if (newIntent == 3) return 3;
            return 0;
        }

        // ═══════════════════════════════════════════════════════
        // FIND ACTIVE MANTIS LORD — GERARCHIA FASE1 → FASE2
        // (RIPRISTINATO DAL VECCHIO CODICE + TIEBREAKER DISTANZA)
        //
        // Fase 1: Se "Mantis Lord" (la principale) è viva → LUI è il target.
        //         Le sorelle non esistono ancora, non confondersi.
        //
        // Fase 2: Sorelle attive. Sort per:
        //         1. Y bassa (= a terra, = attaccabile)
        //         2. Tiebreaker: distanza dal player (più vicina = più pericolosa)
        // ═══════════════════════════════════════════════════════
        private GameObject FindActiveMantisLord(Vector3 playerPos)
        {
            // ─── FASE 1: Boss principale ha priorità assoluta ───
            GameObject mainLord = GameObject.Find("Mantis Lord");
            if (mainLord != null && mainLord.activeSelf)
            {
                var hm = mainLord.GetComponent<HealthManager>();
                if (hm != null && hm.hp > 0) return mainLord;
            }

            // ─── FASE 2: Cerca le sorelle ───
            string[] sisters = { "Mantis Lord S1", "Mantis Lord S2" };
            List<GameObject> activeSisters = new List<GameObject>();

            foreach (var name in sisters)
            {
                var obj = GameObject.Find(name);
                if (obj != null && obj.activeSelf)
                {
                    var hm = obj.GetComponent<HealthManager>();
                    if (hm != null && hm.hp > 0)
                    {
                        // Filtra proiettili con nome simile
                        if (obj.name.ToLower().Contains("shot") || obj.name.ToLower().Contains("scythe"))
                            continue;

                        var fsm = obj.GetComponent<PlayMakerFSM>();
                        if (fsm != null && fsm.FsmName == "Mantis Lord")
                            activeSisters.Add(obj);
                    }
                }
            }

            if (activeSisters.Count == 0) return null;
            if (activeSisters.Count == 1) return activeSisters[0];

            // Sort: Y bassa prima, poi distanza dal player come tiebreaker
            activeSisters.Sort((a, b) =>
            {
                float yDiff = Mathf.Abs(a.transform.position.y - b.transform.position.y);

                // Se hanno Y molto diversa (>2 unità), la più bassa vince
                // (a terra = attaccabile, in aria = sta facendo un attacco aereo)
                if (yDiff > 2.0f)
                {
                    return a.transform.position.y.CompareTo(b.transform.position.y);
                }

                // Stessa altezza → la più vicina al player
                float distA = Vector2.Distance(playerPos, a.transform.position);
                float distB = Vector2.Distance(playerPos, b.transform.position);
                return distA.CompareTo(distB);
            });

            return activeSisters[0];
        }

        // ═══════════════════════════════════════════════════════
        // KILL COUNTING
        // ═══════════════════════════════════════════════════════
        private int CountDeadMantisLords()
        {
            int killed = 0;

            var boss1 = GameObject.Find("Mantis Lord");
            var boss2 = GameObject.Find("Mantis Lord S1");
            var boss3 = GameObject.Find("Mantis Lord S2");

            if (IsDead(boss1)) killed++;

            int s1Hp = GetHealth(boss2);
            int s2Hp = GetHealth(boss3);
            bool s1Exists = (boss2 != null && boss2.activeSelf);
            bool s2Exists = (boss3 != null && boss3.activeSelf);

            if (s1Exists || s2Exists)
            {
                if (killed == 0) killed = 1;
                phase2Detected = true;

                if (s1Exists && s1Hp > 0) lastKnownS1Hp = s1Hp;
                if (s2Exists && s2Hp > 0) lastKnownS2Hp = s2Hp;

                bool s1Dead = IsDead(boss2);
                bool s2Dead = IsDead(boss3);

                if (lastKnownS1Hp > 0 && s1Hp <= 0) s1Dead = true;
                if (lastKnownS2Hp > 0 && s2Hp <= 0) s2Dead = true;

                if (s1Dead) confirmedPhase2Kills = Mathf.Max(confirmedPhase2Kills, 1);
                if (s2Dead) confirmedPhase2Kills = Mathf.Max(confirmedPhase2Kills, 1);
                if (s1Dead && s2Dead) confirmedPhase2Kills = 2;

                killed += confirmedPhase2Kills;

                if (confirmedPhase2Kills > 0)
                    DesktopLogger.Log($"[Kill] Phase2 kills={confirmedPhase2Kills} | S1: hp={s1Hp} dead={s1Dead} | S2: hp={s2Hp} dead={s2Dead}");
            }
            else if (phase2Detected)
            {
                killed = 3;
            }

            return killed;
        }

        // ═══════════════════════════════════════════════════════
        // HEALTH HELPERS
        // ═══════════════════════════════════════════════════════
        private float CalculateTotalMantisHealth()
        {
            string[] lords = { "Mantis Lord", "Mantis Lord S1", "Mantis Lord S2" };
            float totalHp = 0;
            bool foundAny = false;

            foreach (var name in lords)
            {
                var obj = GameObject.Find(name);
                if (obj != null)
                {
                    var hm = obj.GetComponent<HealthManager>();
                    if (hm != null && hm.hp > 0)
                    {
                        totalHp += (float)hm.hp;
                        foundAny = true;
                    }
                }
            }
            if (!foundAny) return -1f;
            return totalHp;
        }

        private int GetHealth(GameObject obj)
        {
            if (obj == null) return 0;
            var hm = obj.GetComponent<HealthManager>();
            return (hm != null) ? hm.hp : 0;
        }

        private bool IsDead(GameObject obj)
        {
            if (obj == null || !obj.activeSelf) return true;
            return GetHealth(obj) <= 0;
        }

        // ═══════════════════════════════════════════════════════
        // HAZARDS
        // ═══════════════════════════════════════════════════════
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

                if ((damageHero != null && isStatic) || name.Contains("spike") || name.Contains("thorn"))
                {
                    isSpike = true;
                    hazardType = "spike";
                }

                if (name.Contains("shot") || name.Contains("scythe") || name.Contains("boomerang") ||
                   (damageHero != null && !isStatic))
                {
                    isProjectile = true;
                    hazardType = "projectile";
                }

                if (hit.gameObject.layer == LayerMask.NameToLayer("Enemies"))
                {
                    if (!isSpike && !isProjectile)
                    {
                        // Ignora mantidi sedute sui troni (Y molto alta rispetto al player)
                        // MA solo se non siamo in fase 2 (in fase 2 attaccano dall'alto)
                        bool isPhase2 = phase2Detected || (prevMantisLordsKilled > 0);
                        if (!isPhase2 && name.Contains("mantis") && hit.transform.position.y > playerPos.y + 5.0f)
                        {
                            continue;
                        }
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

        // ═══════════════════════════════════════════════════════
        // DEFAULT STATE
        // ═══════════════════════════════════════════════════════
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