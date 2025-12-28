using Modding;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;

namespace SyntheticSoulMod
{
    public class SyntheticSoulMod : Mod
    {
        private SocketServer socketServer;
        private GameStateCapture stateCapture;
        private InputHandler inputHandler;

        // Reflection Fields
        private FieldInfo moveInputField;
        
        // Timer per azioni
        private float lastJumpTime = 0f;
        private float lastAttackTime = 0f;
        private float lastDashTime = 0f;

        // Loop dati
        private int updateCounter = 0;
        private int lastHP = 9;
        private int damageCounter = 0;

        public override string GetVersion() => "2.0.0";

        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            try
            {
                inputHandler = new InputHandler();
                stateCapture = new GameStateCapture();
                socketServer = new SocketServer(8888);

                // 1. SETUP REFLECTION ROBUSTO
                // Proviamo a trovare il campo 'move_input' in tutti i modi possibili
                var flags = BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public;
                moveInputField = typeof(HeroController).GetField("move_input", flags);
                if (moveInputField == null) moveInputField = typeof(HeroController).GetField("moveInput", flags);
                
                if (moveInputField == null)
                    Modding.Logger.Log("❌ [SyntheticSoul] FATAL ERROR: Impossibile trovare 'move_input'. Il movimento non funzionerà.");
                else
                    Modding.Logger.Log("✅ [SyntheticSoul] Campo movimento agganciato correttamente.");

                // 2. CALLBACK SOCKET
                socketServer.ActionCallback = action =>
                {
                    if (inputHandler != null) inputHandler.ExecuteAction(action);
                };

                // 3. START SERVER
                Thread socketThread = new Thread(() => socketServer.Start(stateCapture)) { IsBackground = true };
                socketThread.Start();

                // 4. HOOKS
                // Usiamo LookForInput perché avviene PRIMA della fisica nel ciclo del gioco.
                On.HeroController.LookForInput += OnLookForInput;
                
                // Usiamo Update standard per azioni one-shot e invio dati
                ModHooks.HeroUpdateHook += OnHeroUpdateLoop;

                Modding.Logger.Log("Synthetic Soul 2.0.0 - Caricato");
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[Synthetic Soul] ❌ Init Error: {e.Message}");
            }
        }

        // --- CUORE DEL SISTEMA DI MOVIMENTO ---
        private void OnLookForInput(On.HeroController.orig_LookForInput orig, HeroController self)
        {
            // A. Lascia che il gioco legga la tastiera/controller reale
            // Questo imposterà move_input a 0 (perché non tocchi nulla)
            orig(self);

            // B. Sovrascrivi immediatamente con i dati dell'IA
            if (inputHandler != null)
            {
                float aiInput = inputHandler.GetHorizontalInput();

                // Se l'IA sta premendo una direzione (non è 0)
                if (Mathf.Abs(aiInput) > 0.1f)
                {
                    // 1. Inietta il valore nella variabile interna che controlla la fisica
                    if (moveInputField != null)
                    {
                        moveInputField.SetValue(self, aiInput);
                    }

                    // 2. Forza la direzione visiva (risolve il problema che non si gira)
                    if (aiInput > 0)
                    {
                        self.FaceRight();
                    }
                    else
                    {
                        self.FaceLeft();
                    }
                }
            }
        }

        // --- GESTIONE AZIONI E DATI ---
        private void OnHeroUpdateLoop()
        {
            HeroController hero = HeroController.instance;
            if (hero == null) return;

            // Esecuzione Azioni One-Shot
            if (inputHandler != null)
            {
                if (inputHandler.jumpRequested && Time.time - lastJumpTime > 0.4f) {
                    CallMethod(hero, "Jump"); lastJumpTime = Time.time; inputHandler.jumpRequested = false;
                }
                if (inputHandler.attackRequested && Time.time - lastAttackTime > 0.4f) {
                    CallMethod(hero, "DoAttack"); lastAttackTime = Time.time; inputHandler.attackRequested = false;
                }
                if (inputHandler.dashRequested && Time.time - lastDashTime > 0.8f) {
                    CallMethod(hero, "Dash"); lastDashTime = Time.time; inputHandler.dashRequested = false;
                }
                if (inputHandler.focusRequested) {
                    CallMethod(hero, "Focus"); inputHandler.focusRequested = false;
                }
            }

            // Invio Dati IA (ogni 5 frame)
            if (++updateCounter % 5 == 0)
            {
                CaptureAndSendState(hero);
            }
        }

        private void CallMethod(object instance, string methodName)
        {
            try {
                MethodInfo mi = typeof(HeroController).GetMethod(methodName, BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                if (mi != null) mi.Invoke(instance, null);
            } catch { }
        }

        private void CaptureAndSendState(HeroController hero)
        {
            try
            {
                int hp = PlayerData.instance.health;
                int maxHp = PlayerData.instance.maxHealth;
                int soul = PlayerData.instance.MPCharge;
                int maxSoul = PlayerData.instance.maxMP;
                Vector3 pos = hero.transform.position;
                Vector2 vel = hero.GetComponent<Rigidbody2D>().velocity;

                bool tookDamage = false;
                if (hp < lastHP) { tookDamage = true; damageCounter = 90; }
                lastHP = hp;
                if (damageCounter > 0) damageCounter--;

                List<GameStateCapture.Enemy> enemies = new List<GameStateCapture.Enemy>();
                HealthManager[] allHealths = UnityEngine.Object.FindObjectsOfType<HealthManager>();
                foreach (HealthManager hm in allHealths)
                {
                    if (hm.gameObject == hero.gameObject) continue;
                    float dist = Vector3.Distance(pos, hm.transform.position);
                    if (dist < 20f)
                    {
                        enemies.Add(new GameStateCapture.Enemy {
                            Type = hm.gameObject.name, X = hm.transform.position.x, Y = hm.transform.position.y, Distance = dist, HP = hm.hp
                        });
                    }
                }

                stateCapture.UpdatePlayerState(
                    pos, hp, maxHp, vel, soul, maxSoul, enemies,
                    hero.cState.onGround, hero.cState.touchingWall, hero.cState.touchingWall,
                    tookDamage, damageCounter > 0
                );
            }
            catch { }
        }
    }
}