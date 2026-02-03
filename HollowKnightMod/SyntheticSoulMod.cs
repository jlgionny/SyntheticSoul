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
        private const int DEFAULT_PORT = 5555;
        private int PORT;
        private const float UPDATE_INTERVAL = 0.05f;
        private float timeSinceLastUpdate = 0f;
        private bool wasConnected = false;

        // ============ TRAINING SPEED ============
        // NOTA: Velocità 1x per evitare ritardi con l'agente
        private const float TRAINING_TIMESCALE = 1.0f;
        private bool trainingSpeedActive = false;
        private bool autoSpawnTriggered = false;
        private float originalFixedDeltaTime = 0.02f;
        private float lastTimeScaleCheck = 0f;
        private const float TIMESCALE_CHECK_INTERVAL = 0.1f;

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

        // --- AGGIUNTA NUOVA ---
        private int lastHazardTypeDetected = 0;
        // ---------------------

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

        public override string GetVersion() => "9.5.0.0";

        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            Log("Initializing SyntheticSoul Mod v9.5.0 (AUTO-SPAWN + 1X SPEED)...");
            DesktopLogger.Log("=== SYNTHETIC SOUL MOD v9.5.0 - AUTO-SPAWN + 1X SPEED ===");

            PORT = DEFAULT_PORT;
            try
            {
                string gameDir = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
                DesktopLogger.Log($"[Config] DLL location: {gameDir}");

                string[] possiblePaths = new string[]
                {
                    System.IO.Path.Combine(gameDir, "..", "..", "..", "..", "synthetic_soul_port.txt"),
                    System.IO.Path.Combine(gameDir, "..", "..", "..", "synthetic_soul_port.txt"),
                    System.IO.Path.Combine(gameDir, "..", "..", "synthetic_soul_port.txt"),
                    System.IO.Path.Combine(gameDir, "..", "synthetic_soul_port.txt"),
                    System.IO.Path.Combine(gameDir, "synthetic_soul_port.txt"),
                    "synthetic_soul_port.txt"
                };

                foreach (string path in possiblePaths)
                {
                    try
                    {
                        string fullPath = System.IO.Path.GetFullPath(path);
                        DesktopLogger.Log($"[Config] Checking: {fullPath}");
                        if (System.IO.File.Exists(fullPath))
                        {
                            string content = System.IO.File.ReadAllText(fullPath).Trim();
                            DesktopLogger.Log($"[Config] Found file with content: '{content}'");
                            if (int.TryParse(content, out int filePort))
                            {
                                PORT = filePort;
                                DesktopLogger.Log($"[Config] ✓ Using port from file: {PORT} ({fullPath})");
                                break;
                            }
                        }
                    }
                    catch (Exception pathEx)
                    {
                        DesktopLogger.Log($"[Config] Path check error: {pathEx.Message}");
                    }
                }

                if (PORT == DEFAULT_PORT)
                {
                    string portEnv = Environment.GetEnvironmentVariable("SYNTHETIC_SOUL_PORT");
                    if (!string.IsNullOrEmpty(portEnv) && int.TryParse(portEnv, out int envPort))
                    {
                        PORT = envPort;
                        DesktopLogger.Log($"[Config] Using port from env: {PORT}");
                    }
                }

                if (PORT == DEFAULT_PORT)
                {
                    DesktopLogger.Log($"[Config] Using default port: {PORT}");
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"[Config] Error reading port config: {e.Message}");
                PORT = DEFAULT_PORT;
            }

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

            originalFixedDeltaTime = Time.fixedDeltaTime;
            Log("SyntheticSoul Mod ready!");
            DesktopLogger.Log("✓ Boss FSM monitoring active");
            DesktopLogger.Log("✓ Mantis Lords multi-kill tracking enabled");
            DesktopLogger.Log($"✓ Training speed: {TRAINING_TIMESCALE}x when connected");
            DesktopLogger.Log($"✓ Physics-safe timeScale implementation");
            DesktopLogger.Log($"✓ Listening on port {PORT}");
            DesktopLogger.Log("✓ Auto-spawn to GG_Mantis_Lords on connect");
        }

        private void SetTrainingSpeed(bool enable)
        {
            if (enable)
            {
                if (!trainingSpeedActive)
                {
                    trainingSpeedActive = true;
                    if (originalFixedDeltaTime <= 0f)
                        originalFixedDeltaTime = Time.fixedDeltaTime;

                    Time.timeScale = TRAINING_TIMESCALE;
                    DesktopLogger.Log($"[Speed] TimeScale = {TRAINING_TIMESCALE}x, FixedDeltaTime = {Time.fixedDeltaTime}");
                }
            }
            else
            {
                if (trainingSpeedActive)
                {
                    trainingSpeedActive = false;
                    Time.timeScale = 1.0f;
                    Time.fixedDeltaTime = originalFixedDeltaTime;
                    DesktopLogger.Log("[Speed] TimeScale reset to 1x");
                }
            }
        }

        private void EnforceTrainingSpeed()
        {
            if (!trainingSpeedActive) return;
            if (communicator == null || !communicator.IsConnected) return;

            if (Math.Abs(Time.timeScale - TRAINING_TIMESCALE) > 0.01f)
            {
                Time.timeScale = TRAINING_TIMESCALE;
            }
        }
        private void OnSceneChanged(Scene from, Scene to)
        {
            if (sceneChangeHandled)
            {
                return;
            }

            sceneChangeHandled = true;

            if (trainingSpeedActive && communicator != null && communicator.IsConnected)
            {
                SetTrainingSpeed(true);
                DesktopLogger.Log($"[Scene] Training speed maintained at {TRAINING_TIMESCALE}x");
            }
            else if (trainingSpeedActive)
            {
                SetTrainingSpeed(false);
            }

            currentScene = to.name;
            episodeEnded = false;
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

        private IEnumerator RestoreHeroAfterReload()
        {
            DesktopLogger.Log("[Restore] ═══════════════════════════════════════");
            DesktopLogger.Log("[Restore] ═══ SCENE RESTORATION v9.3.0 ═══");
            DesktopLogger.Log("[Restore] ═══════════════════════════════════════");

            ignoreDamageUntilReady = true;
            DesktopLogger.Log("[Restore] Damage tracking DISABLED");

            yield return new WaitForSeconds(0.1f);

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

            yield return new WaitForSeconds(0.05f);
            CleanupDuplicateHeroesOnly();
            CleanupDeathObjects();

            DesktopLogger.Log("[Restore] Waiting for Knight entrance animation to complete...");
            timeout = 0f;
            while (hero.cState.transitioning && timeout < 6f)
            {
                yield return new WaitForSeconds(0.1f);
                timeout += 0.1f;
            }

            DesktopLogger.Log($"[Restore] Transition complete. Hero transitioning: {hero.cState.transitioning}");

            yield return new WaitForEndOfFrame();
            yield return new WaitForSeconds(0.1f);

            PlayMakerFSM bossFSM = FindBossFSM();
            if (bossFSM != null)
            {
                DesktopLogger.Log($"[Restore] Boss FSM found: {bossFSM.gameObject.name} - FSM: {bossFSM.FsmName}");
                DesktopLogger.Log($"[Restore] Current boss state: {bossFSM.ActiveStateName}");

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

            var spriteRenderer = hero.GetComponent<SpriteRenderer>();
            if (spriteRenderer != null)
            {
                spriteRenderer.enabled = true;
                spriteRenderer.color = Color.white;
            }

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

            yield return new WaitForSeconds(0.2f);
            ignoreDamageUntilReady = false;
            DesktopLogger.Log("[Restore] Damage tracking ENABLED");

            lock (damageLock)
            {
                damageTakenSinceLastUpdate = 0;

                // --- AGGIUNTA NUOVA ---
                lastHazardTypeDetected = 0;
                // ---------------------
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

        private IEnumerator AutoSpawnToMantisArena()
        {
            DesktopLogger.Log("[AutoSpawn] ═══════════════════════════════════════");
            DesktopLogger.Log("[AutoSpawn] ═══ TELEPORTING TO MANTIS LORDS ═══");
            DesktopLogger.Log("[AutoSpawn] ═══════════════════════════════════════");

            yield return new WaitForSeconds(0.5f);

            if (GameManager.instance == null)
            {
                DesktopLogger.LogError("[AutoSpawn] GameManager not available!");
                autoSpawnTriggered = false;
                yield break;
            }

            var pd = PlayerData.instance;
            if (pd != null)
            {
                pd.bossRushMode = true;
                pd.SetBool("atBench", false);
                DesktopLogger.Log("[AutoSpawn] PlayerData configured for boss rush");
            }

            isReloading = true;
            ignoreDamageUntilReady = true;
            SetTrainingSpeed(true);
            DesktopLogger.Log("[AutoSpawn] Loading GG_Mantis_Lords...");

            sceneChangeHandled = false;
            UnityEngine.SceneManagement.SceneManager.LoadScene("GG_Mantis_Lords");
            DesktopLogger.Log("[AutoSpawn] Scene load initiated!");
        }

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
                    DesktopLogger.Log($"[Cleanup] ✓ Removed {cleaned} duplicate(s)");
                else
                    DesktopLogger.Log("[Cleanup] ✓ No duplicates found");
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

        private IEnumerator HandleCleanReload(bool isDeath)
        {
            isReloading = true;
            ignoreDamageUntilReady = true;

            var gameState = stateExtractor?.ExtractState();
            if (gameState != null)
            {
                lock (damageLock)
                {
                    gameState.damageTaken = damageTakenSinceLastUpdate;

                    // --- AGGIUNTA NUOVA ---
                    gameState.lastHazardType = lastHazardTypeDetected;
                    // ---------------------

                    damageTakenSinceLastUpdate = 0;

                    // --- AGGIUNTA NUOVA ---
                    lastHazardTypeDetected = 0;
                    // ---------------------
                }

                gameState.isDead = isDeath;
                gameState.bossDefeated = !isDeath;
                communicator?.SendState(gameState);
            }

            yield return new WaitForSeconds(0.2f);
            DesktopLogger.Log("[Reload] ═══ RELOADING SCENE ═══");

            if (trainingSpeedActive && communicator != null && communicator.IsConnected)
            {
                SetTrainingSpeed(true);
                DesktopLogger.Log($"[Reload] TimeScale maintained at {TRAINING_TIMESCALE}x");
            }
            else
            {
                SetTrainingSpeed(false);
            }

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

            var pd = PlayerData.instance;
            if (pd != null)
            {
                pd.bossRushMode = true;
                pd.SetBool("atBench", false);
            }

            sceneChangeHandled = false;
            UnityEngine.SceneManagement.SceneManager.LoadScene(sceneToLoad);
        }

        private void OnHealthManagerDie(On.HealthManager.orig_Die orig, HealthManager self,
            float? attackDirection, AttackTypes attackType, bool ignoreEvasion)
        {
            orig(self, attackDirection, attackType, ignoreEvasion);

            if (isTraining && !episodeEnded && isInBossArena && !isReloading && IsBossEnemy(self.gameObject))
            {
                if (currentScene == "GG_Mantis_Lords")
                {
                    int mantisId = self.gameObject.GetInstanceID();

                    if (killedMantisIds.Contains(mantisId))
                        return;

                    killedMantisIds.Add(mantisId);
                    mantisLordsKilled++;
                    DesktopLogger.Log($"[Victory] Mantis Lord defeated ({mantisLordsKilled}/3)");

                    if (mantisLordsKilled >= 3)
                    {
                        DesktopLogger.Log("[Victory] All Mantis Lords defeated - initiating reload");
                        episodeEnded = true;
                        GameManager.instance.StartCoroutine(HandleCleanReload(isDeath: false));
                    }
                }
                else
                {
                    episodeEnded = true;
                    DesktopLogger.Log("[Victory] Boss defeated");
                    GameManager.instance.StartCoroutine(HandleCleanReload(isDeath: false));
                }
            }
        }

        // ============ MODIFICA CRITICA: OnTakeDamage con hazardType ============
        private int OnTakeDamage(int hazardType, int damage)
        {
            if (ignoreDamageUntilReady || isReloading)
            {
                return 0;
            }

            lock (damageLock)
            {
                damageTakenSinceLastUpdate += damage;

                // --- AGGIUNTA NUOVA ---
                // Salviamo il tipo di pericolo comunicato dal gioco
                // 1 = Normal/Enemy, 2 = Spikes/Acid/Environmental
                lastHazardTypeDetected = hazardType;
                // ---------------------
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

                    if (!autoSpawnTriggered && currentScene != "GG_Mantis_Lords")
                    {
                        autoSpawnTriggered = true;
                        DesktopLogger.Log("[AutoSpawn] Teleporting to GG_Mantis_Lords arena...");
                        GameManager.instance.StartCoroutine(AutoSpawnToMantisArena());
                    }

                    SetTrainingSpeed(true);
                }
                else
                {
                    Log("[SyntheticSoul] ✗ Python agent disconnected");

                    if (actionExecutor != null)
                    {
                        actionExecutor.DestroyDevice();
                    }

                    SetTrainingSpeed(false);
                }

                wasConnected = currentlyConnected;
            }

            if (!currentlyConnected)
                return;

            if (actionExecutor != null)
                actionExecutor.Update();

            lastTimeScaleCheck += Time.unscaledDeltaTime;
            if (lastTimeScaleCheck >= TIMESCALE_CHECK_INTERVAL)
            {
                lastTimeScaleCheck = 0f;
                EnforceTrainingSpeed();
            }

            timeSinceLastUpdate += Time.unscaledDeltaTime;
            if (timeSinceLastUpdate >= UPDATE_INTERVAL)
            {
                timeSinceLastUpdate = 0f;
                ProcessAIStep();
            }
        }

        // ============ MODIFICA CRITICA: ProcessAIStep con lastHazardType ============
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

                var gameState = stateExtractor.ExtractState();
                // !!! FIX IMPORTANTE PER MANTIS KILLED !!!
                // Sovrascriviamo il valore dell'extractor con il nostro contatore sicuro
                // che viene aggiornato dall'evento OnHealthManagerDie
                // ========================================================
                gameState.mantisLordsKilled = this.mantisLordsKilled;

                lock (damageLock)
                {
                    gameState.damageTaken = damageTakenSinceLastUpdate;

                    // --- AGGIUNTA NUOVA ---
                    gameState.lastHazardType = lastHazardTypeDetected;

                    // Reset dopo l'invio
                    damageTakenSinceLastUpdate = 0;
                    lastHazardTypeDetected = 0; // Reset a 0, non a 1
                    // ---------------------
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