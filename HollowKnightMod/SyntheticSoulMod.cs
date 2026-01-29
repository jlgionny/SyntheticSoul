using Modding;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using GlobalEnums;
using HutongGames.PlayMaker;
using HutongGames.PlayMaker.Actions;

namespace SyntheticSoulMod
{
    public class SyntheticSoulMod : Mod, ITogglableMod
    {
        private static SyntheticSoulMod _instance;
        private SocketCommunicator communicator;
        private GameStateExtractor stateExtractor;
        private ActionExecutor actionExecutor;
        private bool isTraining = false;
        private HeroController hero;

        private const int PORT = 5555;
        private const float UPDATE_INTERVAL = 0.05f;
        private float timeSinceLastUpdate = 0f;
        private bool wasConnected = false;

        // ============ SCENE STATE VARIABLES ============
        private string currentScene = "";
        private string lastBossScene = "";
        private bool isInBossArena = false;
        private bool episodeEnded = false;
        private bool isReloading = false;
        private bool sceneChangeHandled = false;

        // ============ MANTIS LORDS TRACKING ============
        private int mantisLordsKilled = 0;
        private HashSet<int> killedMantisIds = new HashSet<int>();

        // ============ DAMAGE ACCUMULATOR ============
        private int damageTakenSinceLastUpdate = 0;
        private readonly object damageLock = new object();

        // ============ RELOAD SAFETY ============
        private bool ignoreDamageUntilReady = false;

        public static SyntheticSoulMod Instance
        {
            get
            {
                if (_instance == null)
                    _instance = new SyntheticSoulMod();
                return _instance;
            }
        }

        public SyntheticSoulMod() : base("Synthetic Soul Boss AI")
        {
            _instance = this;
        }

        public override string GetVersion() => "9.3.0.0";

        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            Log("Initializing SyntheticSoul Mod v9.3.0 (NO AUTO-LOAD)...");
            DesktopLogger.Log("=== SYNTHETIC SOUL MOD v9.3.0 - NO AUTO-LOAD ===");

            stateExtractor = new GameStateExtractor();
            actionExecutor = new ActionExecutor();
            communicator = new SocketCommunicator(PORT);

            ModHooks.HeroUpdateHook += OnHeroUpdate;
            ModHooks.AfterTakeDamageHook += OnTakeDamage;
            On.HealthManager.Die += OnHealthManagerDie;
            On.HeroController.Die += OnHeroDeath;
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnSceneChanged;

            Log("SyntheticSoul Mod initialized! Starting persistent server...");

            try
            {
                communicator.StartServer();
                isTraining = true;
                Log("Persistent server started!");
                DesktopLogger.Log("✓ Server ready");
            }
            catch (Exception e)
            {
                LogError($"Failed to start server: {e.Message}");
                DesktopLogger.LogError($"Server start failed: {e.Message}\n{e.StackTrace}");
            }

            Log("SyntheticSoul Mod ready!");
            DesktopLogger.Log("✓ Boss FSM monitoring active");
            DesktopLogger.Log("✓ Mantis Lords multi-kill tracking enabled");
            DesktopLogger.Log("✓ Waiting for Python connection (NO AUTO-LOAD)");
        }

        // ============ SCENE TRACKING ============
        private void OnSceneChanged(Scene from, Scene to)
        {
            if (sceneChangeHandled)
            {
                return;
            }

            sceneChangeHandled = true;

            // Reset timeScale PRIMA di tutto
            if (Time.timeScale != 1f)
            {
                Time.timeScale = 1f;
                DesktopLogger.Log("[Scene] Time.timeScale = 1 (reset)");
            }

            currentScene = to.name;
            episodeEnded = false;

            // Reset contatore Mantis Lords
            mantisLordsKilled = 0;
            killedMantisIds.Clear();
            stateExtractor?.ResetTracking();

            DesktopLogger.Log($"[Scene] Entered: {currentScene} (from: {from.name})");

            if (currentScene.StartsWith("GG_") || IsBossScene(currentScene))
            {
                isInBossArena = true;
                if (IsBossScene(currentScene))
                {
                    lastBossScene = currentScene;
                }

                if (isReloading)
                {
                    // NON bloccare qui - StartCoroutine gestirà il timing
                    GameManager.instance.StartCoroutine(RestoreHeroAfterReload());
                }
            }
            else
            {
                isInBossArena = false;
                isReloading = false;
            }

            GameManager.instance.StartCoroutine(ResetSceneChangeFlag());
        }

        private IEnumerator ResetSceneChangeFlag()
        {
            yield return new WaitForSeconds(0.5f);
            sceneChangeHandled = false;
        }

        // ============ RESTORE HERO (WAIT FOR NATURAL SEQUENCE + BOSS FSM) ============
        private IEnumerator RestoreHeroAfterReload()
        {
            DesktopLogger.Log("[Restore] ═══════════════════════════════════════");
            DesktopLogger.Log("[Restore] ═══ SCENE RESTORATION v9.3.0 ═══");
            DesktopLogger.Log("[Restore] ═══════════════════════════════════════");

            // STEP 1: BLOCCA DANNI IMMEDIATAMENTE
            ignoreDamageUntilReady = true;
            DesktopLogger.Log("[Restore] Damage tracking DISABLED");

            // STEP 2: Aspetta che la scena inizi a caricarsi
            yield return new WaitForSeconds(0.1f);

            // STEP 3: ATTENDI HERO SPAWN
            float timeout = 0f;
            while (HeroController.instance == null && timeout < 3f)
            {
                yield return new WaitForSeconds(0.05f);
                timeout += 0.05f;
            }

            if (HeroController.instance == null)
            {
                DesktopLogger.LogError("[Restore] Hero never spawned!");
                isReloading = false;
                ignoreDamageUntilReady = false;
                yield break;
            }

            var hero = HeroController.instance;
            DesktopLogger.Log($"[Restore] ✓ Hero found: {hero.gameObject.name}");

            // STEP 4: Pulisci duplicati e oggetti di morte SENZA toccare l'hero corrente
            yield return new WaitForSeconds(0.05f);
            CleanupDuplicateHeroesOnly();
            CleanupDeathObjects();

            // STEP 5: ASPETTA CHE L'ANIMAZIONE DI ENTRATA FINISCA
            DesktopLogger.Log("[Restore] Waiting for Knight entrance animation to complete...");
            timeout = 0f;
            while (hero.cState.transitioning && timeout < 6f)
            {
                yield return new WaitForSeconds(0.1f);
                timeout += 0.1f;
            }

            DesktopLogger.Log($"[Restore] Transition complete. Hero transitioning: {hero.cState.transitioning}");

            // STEP 6: ASPETTA UN FRAME EXTRA PER IL POSIZIONAMENTO FINALE
            yield return new WaitForEndOfFrame();
            yield return new WaitForSeconds(0.1f);

            // STEP 7: TROVA E MONITORA LA FSM DEL BOSS
            PlayMakerFSM bossFSM = FindBossFSM();
            if (bossFSM != null)
            {
                DesktopLogger.Log($"[Restore] Boss FSM found: {bossFSM.gameObject.name} - FSM: {bossFSM.FsmName}");
                DesktopLogger.Log($"[Restore] Current boss state: {bossFSM.ActiveStateName}");

                // Aspetta che il boss completi l'intro
                timeout = 0f;
                string previousState = bossFSM.ActiveStateName;
                bool bossIntroStarted = false;

                while (timeout < 5f)
                {
                    yield return new WaitForSeconds(0.1f);
                    timeout += 0.1f;

                    string currentState = bossFSM.ActiveStateName;
                    if (currentState != previousState)
                    {
                        DesktopLogger.Log($"[Restore] Boss state changed: {previousState} -> {currentState}");
                        previousState = currentState;
                        bossIntroStarted = true;
                    }

                    if (bossIntroStarted && !IsIntroState(currentState))
                    {
                        DesktopLogger.Log($"[Restore] ✓ Boss intro complete! Active state: {currentState}");
                        break;
                    }
                }

                if (timeout >= 5f)
                    DesktopLogger.Log("[Restore] ⚠ Boss intro timeout - proceeding anyway");
            }
            else
            {
                DesktopLogger.Log("[Restore] No boss FSM found - using fixed delay");
                yield return new WaitForSeconds(2.5f);
            }

            // STEP 8: ORA possiamo resettare la salute
            DesktopLogger.Log("[Restore] Setting hero health...");
            var pd = PlayerData.instance;
            if (pd != null)
            {
                pd.health = pd.maxHealth;
                pd.MPCharge = 0;
                pd.MPReserve = 0;
                pd.isInvincible = false;
                DesktopLogger.Log($"[Reset] PlayerData: HP={pd.health}/{pd.maxHealth}");
            }

            hero.MaxHealth();
            if (hero.cState.dead)
            {
                hero.cState.dead = false;
                DesktopLogger.Log("[Reset] Cleared dead flag");
            }

            // Reset visuals
            var spriteRenderer = hero.GetComponent<SpriteRenderer>();
            if (spriteRenderer != null)
            {
                spriteRenderer.enabled = true;
                spriteRenderer.color = Color.white;
            }

            // STEP 9: Verifica controllo input
            if (!hero.acceptingInput)
            {
                DesktopLogger.Log("[Restore] ⚠ Forcing input control...");
                hero.RegainControl();
                hero.AcceptInput();

                for (int i = 0; i < 3; i++)
                {
                    if (hero.acceptingInput) break;
                    hero.AcceptInput();
                    yield return new WaitForSeconds(0.05f);
                }
            }
            else
            {
                DesktopLogger.Log("[Restore] ✓ Hero already has input control");
            }

            // STEP 10: RIATTIVA DANNI
            yield return new WaitForSeconds(0.2f);
            ignoreDamageUntilReady = false;
            DesktopLogger.Log("[Restore] Damage tracking ENABLED");

            lock (damageLock)
            {
                damageTakenSinceLastUpdate = 0;
            }

            yield return new WaitForSeconds(0.1f);
            isReloading = false;

            DesktopLogger.Log($"[Restore] ✓ Hero HP: {PlayerData.instance.health}/{PlayerData.instance.maxHealth}");
            DesktopLogger.Log($"[Restore] ✓ Accepting Input: {hero.acceptingInput}");
            DesktopLogger.Log($"[Restore] ✓ Hero dead state: {hero.cState.dead}");
            DesktopLogger.Log($"[Restore] ✓ Hero transitioning: {hero.cState.transitioning}");
            DesktopLogger.Log("[Restore] ═══════════════════════════════════════");
            DesktopLogger.Log("[Restore] ✓✓✓ RESTORATION COMPLETE ✓✓✓");
            DesktopLogger.Log("[Restore] ═══════════════════════════════════════");
        }

        // ============ TROVA LA FSM DEL BOSS NELLA SCENA ============
        private PlayMakerFSM FindBossFSM()
        {
            try
            {
                HealthManager[] healthManagers = GameObject.FindObjectsOfType<HealthManager>();
                foreach (var hm in healthManagers)
                {
                    if (hm.hp < 100) continue;

                    PlayMakerFSM[] fsms = hm.gameObject.GetComponents<PlayMakerFSM>();
                    foreach (var fsm in fsms)
                    {
                        string fsmName = fsm.FsmName.ToLower();
                        if (fsmName.Contains("control") ||
                            fsmName.Contains("boss") ||
                            fsmName.Contains("attack") ||
                            fsmName == "mantis" ||
                            fsmName == "hornet" ||
                            fsmName == "mawlek")
                        {
                            DesktopLogger.Log($"[FSM] Found potential boss FSM: {fsm.gameObject.name}.{fsm.FsmName}");
                            return fsm;
                        }
                    }
                }

                string[] bossObjectNames = new string[]
                {
                    "Mantis Lord", "Hornet Boss", "Mawlek Body",
                    "False Knight", "Mega Moss Charger", "Hive Knight",
                    "Dung Defender", "Traitor Lord", "Giant Buzzer"
                };

                foreach (var bossName in bossObjectNames)
                {
                    GameObject bossObj = GameObject.Find(bossName);
                    if (bossObj != null)
                    {
                        PlayMakerFSM fsm = bossObj.GetComponent<PlayMakerFSM>();
                        if (fsm != null)
                        {
                            DesktopLogger.Log($"[FSM] Found boss by name: {bossObj.name}");
                            return fsm;
                        }
                    }
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"[FSM] Error finding boss: {e.Message}");
            }

            return null;
        }

        private bool IsIntroState(string stateName)
        {
            if (string.IsNullOrEmpty(stateName)) return false;
            string state = stateName.ToLower();
            return state.Contains("idle") ||
                   state.Contains("sleep") ||
                   state.Contains("roar") ||
                   state.Contains("intro") ||
                   state.Contains("wake") ||
                   state.Contains("init") ||
                   state == "start";
        }

        // ============ CLEANUP ============
        private void CleanupDuplicateHeroesOnly()
        {
            try
            {
                var currentHero = HeroController.instance;
                if (currentHero == null)
                {
                    DesktopLogger.Log("[Cleanup] No current hero, skipping cleanup");
                    return;
                }

                GameObject currentHeroGO = currentHero.gameObject;
                GameObject[] allRootObjects = GetAllRootGameObjects();
                int cleaned = 0;

                foreach (var obj in allRootObjects)
                {
                    if (obj == null || obj == currentHeroGO) continue;

                    string name = obj.name.ToLower();
                    if (name.Contains("(clone)") && (name.Contains("knight") || name.Contains("hero")))
                    {
                        DesktopLogger.Log($"[Cleanup] Removing clone: {obj.name}");
                        GameObject.Destroy(obj);
                        cleaned++;
                        continue;
                    }

                    if (name == "knight" && obj != currentHeroGO)
                    {
                        var heroCtrl = obj.GetComponent<HeroController>();
                        if (heroCtrl != null && heroCtrl != currentHero)
                        {
                            DesktopLogger.Log($"[Cleanup] Removing duplicate Knight with HeroController");
                            GameObject.Destroy(obj);
                            cleaned++;
                        }
                    }
                }

                if (cleaned > 0)
                {
                    DesktopLogger.Log($"[Cleanup] ✓ Removed {cleaned} duplicate(s)");
                }
                else
                {
                    DesktopLogger.Log("[Cleanup] ✓ No duplicates found");
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"[Cleanup] Error: {e.Message}");
            }
        }

        private GameObject[] GetAllRootGameObjects()
        {
            List<GameObject> allObjects = new List<GameObject>();
            allObjects.AddRange(UnityEngine.SceneManagement.SceneManager.GetActiveScene().GetRootGameObjects());

            try
            {
                GameObject temp = new GameObject("TempDDOL");
                GameObject.DontDestroyOnLoad(temp);
                UnityEngine.SceneManagement.Scene dontDestroyOnLoadScene = temp.scene;
                GameObject.Destroy(temp);

                if (dontDestroyOnLoadScene.IsValid())
                {
                    allObjects.AddRange(dontDestroyOnLoadScene.GetRootGameObjects());
                }
            }
            catch { }

            return allObjects.ToArray();
        }

        private void CleanupDeathObjects()
        {
            try
            {
                string[] deathObjectNames = new string[]
                {
                    "Hero Death",
                    "Hero_Death_Anim",
                    "Hero Death Anim",
                    "Knight Death",
                    "death_respawn_marker",
                    "Corpse"
                };

                int cleaned = 0;
                GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();

                foreach (var obj in allObjects)
                {
                    if (obj == null) continue;

                    foreach (var deathName in deathObjectNames)
                    {
                        if (obj.name.Contains(deathName))
                        {
                            GameObject.Destroy(obj);
                            cleaned++;
                            break;
                        }
                    }
                }

                if (cleaned > 0)
                {
                    DesktopLogger.Log($"[Cleanup] ✓ Removed {cleaned} death object(s)");
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"[Cleanup] Error: {e.Message}");
            }
        }

        // ============ DEATH HANDLER ============
        private IEnumerator OnHeroDeath(On.HeroController.orig_Die orig, HeroController self)
        {
            if (ignoreDamageUntilReady || isReloading)
            {
                DesktopLogger.Log("[Death] Blocked during reload");
                yield break;
            }

            if (isTraining && !episodeEnded && isInBossArena)
            {
                episodeEnded = true;
                DesktopLogger.Log("[Death] Hero died - initiating reload");
                yield return new WaitForSeconds(0.5f);
                yield return GameManager.instance.StartCoroutine(HandleCleanReload(isDeath: true));
            }
            else
            {
                yield return orig(self);
            }
        }

        // ============ HANDLE CLEAN RELOAD (MODIFIED) ============
        private IEnumerator HandleCleanReload(bool isDeath)
        {
            isReloading = true;
            ignoreDamageUntilReady = true;

            // Invia stato finale
            var gameState = stateExtractor?.ExtractState();
            if (gameState != null)
            {
                lock (damageLock)
                {
                    gameState.damageTaken = damageTakenSinceLastUpdate;
                    damageTakenSinceLastUpdate = 0;
                }
                gameState.isDead = isDeath;
                gameState.bossDefeated = !isDeath;
                communicator?.SendState(gameState);
            }

            yield return new WaitForSeconds(0.2f);
            DesktopLogger.Log("[Reload] ═══ RELOADING SCENE ═══");

            // Mantieni time scale normale
            Time.timeScale = 1f;

            // Ricarica la scena corrente se è una boss arena
            string currentSceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            string sceneToLoad = currentSceneName;

            if (currentSceneName.StartsWith("GG_") &&
                currentSceneName != "GG_Workshop" &&
                currentSceneName != "GG_Atrium" &&
                currentSceneName != "GG_Waterways")
            {
                DesktopLogger.Log($"[Reload] Reloading current Godhome arena: {sceneToLoad}");
            }
            else
            {
                DesktopLogger.Log($"[Reload] Reloading: {sceneToLoad}");
            }

            // Prepara PlayerData per evitare bench respawn
            var pd = PlayerData.instance;
            if (pd != null)
            {
                pd.bossRushMode = true;
                pd.SetBool("atBench", false);
            }

            sceneChangeHandled = false;
            UnityEngine.SceneManagement.SceneManager.LoadScene(sceneToLoad);
        }

        // ============ BOSS DEATH HANDLER WITH MANTIS LORDS TRACKING ============
        private void OnHealthManagerDie(On.HealthManager.orig_Die orig, HealthManager self,
            float? attackDirection, AttackTypes attackType, bool ignoreEvasion)
        {
            orig(self, attackDirection, attackType, ignoreEvasion);

            if (isTraining && !episodeEnded && isInBossArena && !isReloading && IsBossEnemy(self.gameObject))
            {
                // Gestione speciale per Mantis Lords
                if (currentScene == "GG_Mantis_Lords")
                {
                    int mantisId = self.gameObject.GetInstanceID();

                    // Evita conteggi duplicati
                    if (killedMantisIds.Contains(mantisId))
                        return;

                    killedMantisIds.Add(mantisId);
                    mantisLordsKilled++;
                    DesktopLogger.Log($"[Victory] Mantis Lord defeated ({mantisLordsKilled}/3)");

                    // Solo dopo aver ucciso tutte e 3 le mantidi, resetta
                    if (mantisLordsKilled >= 3)
                    {
                        DesktopLogger.Log("[Victory] All Mantis Lords defeated - initiating reload");
                        episodeEnded = true;
                        GameManager.instance.StartCoroutine(HandleCleanReload(isDeath: false));
                    }
                }
                else
                {
                    // Per altri boss, comportamento normale
                    episodeEnded = true;
                    DesktopLogger.Log("[Victory] Boss defeated");
                    GameManager.instance.StartCoroutine(HandleCleanReload(isDeath: false));
                }
            }
        }

        private int OnTakeDamage(int hazardType, int damage)
        {
            if (ignoreDamageUntilReady || isReloading)
            {
                return 0;
            }

            lock (damageLock)
            {
                damageTakenSinceLastUpdate += damage;
            }

            return damage;
        }

        private bool IsBossScene(string sceneName)
        {
            string name = sceneName.ToLower();

            if (name.StartsWith("gg_") &&
                name != "gg_workshop" &&
                name != "gg_atrium" &&
                name != "gg_waterways")
            {
                return true;
            }

            return name.Contains("fungus2_15") || name.Contains("fungus1_04") ||
                   name.Contains("deepnest_east") || name.Contains("fungus3_23") ||
                   name.Contains("ruins1_24") || name.Contains("mines_18") ||
                   name.Contains("crossroads_09");
        }

        private bool IsBossEnemy(GameObject enemy)
        {
            if (enemy == null) return false;

            string name = enemy.name.ToLower();
            bool isBossByName = name.Contains("hornet") || name.Contains("mantis") ||
                name.Contains("false knight") || name.Contains("mawlek") ||
                name.Contains("soul master") || name.Contains("broken vessel") ||
                name.Contains("dung defender") || name.Contains("traitor lord") ||
                name.Contains("collector") || name.Contains("god tamer") ||
                name.Contains("hive knight") || name.Contains("ghost warrior") ||
                name.Contains("crystal guardian") || name.Contains("white defender") ||
                name.Contains("lost kin") || name.Contains("grey prince") ||
                name.Contains("nosk") || name.Contains("flukemarm") ||
                name.Contains("massive moss charger") || name.Contains("gruz mother") ||
                name.Contains("vengefly king") || name.Contains("nightmare king") ||
                name.Contains("radiance") || name.Contains("boss") || name.Contains("gruz");

            if (isBossByName) return true;

            var hm = enemy.GetComponent<HealthManager>();
            return hm != null && hm.hp > 50;
        }

        private void OnHeroUpdate()
        {
            if (hero == null)
                hero = HeroController.instance;

            if (!isTraining)
                return;

            bool currentlyConnected = communicator.IsConnected;

            if (currentlyConnected != wasConnected)
            {
                if (currentlyConnected)
                {
                    Log("[SyntheticSoul] ✓ Python agent connected!");
                    DesktopLogger.Log($"=== PYTHON AGENT CONNECTED ===");

                    if (actionExecutor != null)
                        actionExecutor.DestroyDevice();

                    actionExecutor = new ActionExecutor();
                }
                else
                {
                    Log("[SyntheticSoul] ✗ Python agent disconnected");
                    if (actionExecutor != null)
                    {
                        actionExecutor.DestroyDevice();
                    }
                }

                wasConnected = currentlyConnected;
            }

            if (!currentlyConnected)
                return;

            if (actionExecutor != null)
                actionExecutor.Update();

            timeSinceLastUpdate += Time.deltaTime;
            if (timeSinceLastUpdate >= UPDATE_INTERVAL)
            {
                timeSinceLastUpdate = 0f;
                ProcessAIStep();
            }
        }

        private void ProcessAIStep()
        {
            try
            {
                if (isReloading || episodeEnded || ignoreDamageUntilReady)
                    return;

                if (hero == null)
                {
                    hero = HeroController.instance;
                    return;
                }

                if (hero.cState.transitioning || !hero.acceptingInput)
                    return;

                // Estrai stato anche se NON in boss arena (per test movement)
                var gameState = stateExtractor.ExtractState();
                lock (damageLock)
                {
                    gameState.damageTaken = damageTakenSinceLastUpdate;
                    damageTakenSinceLastUpdate = 0;
                }

                communicator.SendState(gameState);
                string action = communicator.ReceiveAction();

                bool canExecute = hero != null &&
                    !hero.cState.dead &&
                    !hero.cState.recoiling &&
                    hero.acceptingInput;

                if (!string.IsNullOrEmpty(action) && action != "IDLE" && canExecute)
                {
                    actionExecutor.ExecuteAction(action, force: false);
                }
            }
            catch (Exception e)
            {
                LogError($"[SyntheticSoul] AI step error: {e.Message}");
            }
        }

        public void Unload()
        {
            if (actionExecutor != null)
            {
                actionExecutor.DestroyDevice();
                actionExecutor = null;
            }

            ModHooks.HeroUpdateHook -= OnHeroUpdate;
            ModHooks.AfterTakeDamageHook -= OnTakeDamage;
            On.HealthManager.Die -= OnHealthManagerDie;
            On.HeroController.Die -= OnHeroDeath;
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= OnSceneChanged;

            if (communicator != null)
                communicator.Close();

            isTraining = false;
        }
    }
}