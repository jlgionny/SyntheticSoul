using Modding;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Classe principale della Mod. Eredita da 'Mod' (API di Hollow Knight).
    /// Gestisce l'inizializzazione, l'accesso ai dati di gioco tramite Reflection
    /// e l'aggiornamento periodico dello stato.
    /// 
    /// NUOVO: Integra l'InputHandler per eseguire azioni ricevute dall'IA.
    /// </summary>
    public class SyntheticSoulMod : Mod
    {
        // Componenti principali
        private SocketServer socketServer;
        private GameStateCapture stateCapture;
        private InputHandler inputHandler;

        // --- CAMPI REFLECTION (Per accedere a variabili private del gioco) ---
        private Type gameManagerType;
        private PropertyInfo gmInstanceProp; // Accesso a GameManager.instance
        private FieldInfo pdField; // Accesso a PlayerData

        // Campi specifici di PlayerData
        private FieldInfo healthField; // HP attuali
        private FieldInfo maxHealthField; // HP massimi (maschere)
        private FieldInfo mpChargeField; // Anime (Soul) attuali
        private FieldInfo maxMPField; // Capienza massima Anime

        // Campi specifici di HeroController (Sensori)
        private FieldInfo touchingWallLField; // Sta toccando muro a sinistra?
        private FieldInfo touchingWallRField; // Sta toccando muro a destra?

        // --- VARIABILI DI STATO LOCALI ---
        private int lastHP = 9; // Per rilevare il cambio di HP
        private int damageCounter = 0; // Contatore per il cooldown post-danno
        private const int DAMAGE_MALUS_DURATION = 90; // Durata del malus (in frame)

        // Restituisce la versione della mod visibile nel menu
        public override string GetVersion() => "1.1.0";

        /// <summary>
        /// Metodo chiamato al caricamento della mod.
        /// Qui inizializziamo il server e colleghiamo gli Hooks di Unity.
        /// </summary>
        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            try
            {
                // 1. Inizializza i componenti logici
                stateCapture = new GameStateCapture();
                socketServer = new SocketServer(8888); // Porta TCP 8888

                // 2. Setup della Reflection per GameManager (per leggere HP e Soul precisi)
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
                {
                    gameManagerType = assembly.GetType("GameManager");
                    if (gameManagerType != null) break;
                }

                if (gameManagerType != null)
                {
                    gmInstanceProp = gameManagerType.GetProperty("instance", BindingFlags.Public | BindingFlags.Static);
                    pdField = gameManagerType.GetField("playerData", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                }

                // 3. Setup della Reflection per HeroController (per i sensori dei muri)
                try
                {
                    touchingWallLField = typeof(HeroController).GetField("touchingWallL", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    touchingWallRField = typeof(HeroController).GetField("touchingWallR", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                }
                catch { /* Ignora se fallisce */ }

                // 4. Inizializza InputHandler dopo il caricamento del gioco
                ModHooks.HeroUpdateHook += InitializeInputHandlerOnce;

                // 5. Configura il callback del SocketServer per eseguire azioni
                socketServer.ActionCallback += (action) =>
                {
                    if (inputHandler != null)
                    {
                        inputHandler.ExecuteAction(action);
                    }
                };

                // 6. Avvia il Server Socket in un thread separato
                Thread socketThread = new Thread(() => socketServer.Start(stateCapture))
                {
                    IsBackground = true
                };
                socketThread.Start();

                // 7. Aggancia il nostro metodo al loop di aggiornamento dell'Eroe
                ModHooks.HeroUpdateHook += OnHeroUpdate;

                Modding.Logger.Log("✅ [Synthetic Soul v1.1.0] Mod caricata - Socket attivo su porta 8888");
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"❌ [Synthetic Soul] Errore Critico Init: {e.Message}");
            }
        }

        /// <summary>
        /// Inizializza l'InputHandler solo una volta (dopo che HeroController è disponibile)
        /// </summary>
        private void InitializeInputHandlerOnce()
        {
            if (inputHandler == null && HeroController.instance != null)
            {
                inputHandler = new InputHandler(HeroController.instance);
                Modding.Logger.Log("✅ [Synthetic Soul] InputHandler inizializzato");
                
                // Rimuovi questo hook per non eseguirlo più
                ModHooks.HeroUpdateHook -= InitializeInputHandlerOnce;
            }
        }

        private int updateCounter = 0;

        /// <summary>
        /// Eseguito ogni frame in cui l'Eroe è attivo.
        /// Raccoglie i dati e li invia a GameStateCapture.
        /// </summary>
        private void OnHeroUpdate()
        {
            // Eseguiamo la logica solo 1 volta ogni 5 frame per ottimizzare
            if (++updateCounter % 5 != 0) return;

            try
            {
                HeroController hero = HeroController.instance;
                if (hero == null) return;

                // --- RACCOLTA DATI FISICI ---
                Vector3 pos = hero.transform.position;
                Rigidbody2D rb = hero.GetComponent<Rigidbody2D>();
                Vector2 vel = rb != null ? rb.velocity : Vector2.zero;

                // Valori di default
                int hp = 9;
                int maxHp = 12;
                int soul = 0;
                int maxSoul = 99;

                // --- LETTURA DATI TRAMITE REFLECTION ---
                try
                {
                    if (gmInstanceProp != null && pdField != null)
                    {
                        object gmInstance = gmInstanceProp.GetValue(null);
                        if (gmInstance != null)
                        {
                            object playerData = pdField.GetValue(gmInstance);
                            if (playerData != null)
                            {
                                Type pdType = playerData.GetType();
                                
                                if (mpChargeField == null)
                                {
                                    mpChargeField = pdType.GetField("MPCharge", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                    maxMPField = pdType.GetField("maxMP", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                    healthField = pdType.GetField("health", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                    maxHealthField = pdType.GetField("maxHealth", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                }

                                if (mpChargeField != null) soul = (int)mpChargeField.GetValue(playerData);
                                if (maxMPField != null) maxSoul = (int)maxMPField.GetValue(playerData);
                                if (healthField != null) hp = (int)healthField.GetValue(playerData);
                                if (maxHealthField != null) maxHp = (int)maxHealthField.GetValue(playerData);
                            }
                        }
                    }
                }
                catch { /* Fallback ai valori di default */ }

                // --- LOGICA DANNI ---
                bool tookDamage = false;
                if (hp < lastHP)
                {
                    tookDamage = true;
                    damageCounter = DAMAGE_MALUS_DURATION;
                    Modding.Logger.Log($"⚠️ [Synthetic Soul] DANNO: {lastHP} -> {hp}");
                }

                lastHP = hp;

                // Decrementa il timer del malus
                if (damageCounter > 0) damageCounter--;

                // --- ANALISI AMBIENTALE ---
                List<GameStateCapture.Enemy> enemies = FindNearbyEnemies(pos, 20f);
                bool groundBelow = CheckGround(vel);
                bool touchingWallL = CheckWallLeft(hero);
                bool touchingWallR = CheckWallRight(hero);

                // --- AGGIORNAMENTO STATO GLOBALE ---
                stateCapture.UpdatePlayerState(
                    pos, hp, maxHp, vel, soul, maxSoul,
                    enemies, groundBelow, touchingWallL, touchingWallR,
                    tookDamage, damageCounter > 0
                );
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"❌ [Synthetic Soul] Errore Loop: {e.Message}");
            }
        }

        /// <summary>
        /// Cerca tutti i nemici (HealthManager) entro un raggio specificato.
        /// </summary>
        private List<GameStateCapture.Enemy> FindNearbyEnemies(Vector3 playerPos, float radius)
        {
            List<GameStateCapture.Enemy> enemies = new List<GameStateCapture.Enemy>();
            try
            {
                HealthManager[] allHealths = UnityEngine.Object.FindObjectsOfType<HealthManager>();
                foreach (HealthManager hm in allHealths)
                {
                    if (hm.gameObject == HeroController.instance.gameObject) continue;
                    
                    Vector3 enemyPos = hm.transform.position;
                    float distance = Vector3.Distance(playerPos, enemyPos);
                    
                    if (distance < radius)
                    {
                        enemies.Add(new GameStateCapture.Enemy
                        {
                            Type = hm.gameObject.name.Replace("(Clone)", "").Trim(),
                            X = enemyPos.x,
                            Y = enemyPos.y,
                            Distance = distance,
                            HP = hm.hp
                        });
                    }
                }

                enemies.Sort((a, b) => a.Distance.CompareTo(b.Distance));
            }
            catch { }
            
            return enemies;
        }

        // Helper per controllare se il giocatore è a terra
        private bool CheckGround(Vector2 velocity) => velocity.y <= 0.1f && velocity.y >= -0.1f;

        // Helper per leggere il campo privato 'touchingWallL'
        private bool CheckWallLeft(HeroController hero)
        {
            try { return touchingWallLField != null && (bool)touchingWallLField.GetValue(hero); }
            catch { return false; }
        }

        // Helper per leggere il campo privato 'touchingWallR'
        private bool CheckWallRight(HeroController hero)
        {
            try { return touchingWallRField != null && (bool)touchingWallRField.GetValue(hero); }
            catch { return false; }
        }
    }
}
