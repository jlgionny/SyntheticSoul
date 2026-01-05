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
        private FieldInfo moveInputField;

        private float lastJumpTime = 0f;
        private float lastAttackTime = 0f;
        private float lastDashTime = 0f;

        private int updateCounter = 0;      
        
        // MODIFICATO: 6 Frame (0.1s a 60fps). 
        // 15 era troppo lento, 1 era troppo veloce. 6 è il bilanciamento giusto.
        private const int FRAME_SKIP = 6;
        
        private int lastHP = -1;
        private bool damageTakenSinceLastSend = false; 

        public override string GetVersion() => "3.4.0-REACTIVE";

        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            Log("Inizializzazione Synthetic Soul (Reactive Fix)...");
            try
            {
                stateCapture = new GameStateCapture();
                inputHandler = new InputHandler();
                socketServer = new SocketServer(8888); 

                socketServer.ActionCallback = (actionString) => 
                {
                    var actionEnum = InputHandler.ParseAction(actionString);
                    inputHandler.ExecuteAction(actionEnum);
                };

                var flags = BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public;
                moveInputField = typeof(HeroController).GetField("move_input", flags);
                if (moveInputField == null) moveInputField = typeof(HeroController).GetField("moveInput", flags);

                Thread serverThread = new Thread(() => socketServer.Start(stateCapture));
                serverThread.IsBackground = true;
                serverThread.Start();

                On.HeroController.LookForInput += OnLookForInput;
                ModHooks.HeroUpdateHook += OnHeroUpdate;

                Log("Synthetic Soul Ready!");
            }
            catch (Exception e) { Log("Init Error: " + e.Message); }
        }

        // --- HOOK MOVIMENTO ---
        private void OnLookForInput(On.HeroController.orig_LookForInput orig, HeroController self)
        {
            orig(self);

            // 1. SICURA DANNI: Se stiamo subendo danno...
            if (self.cState.recoiling || self.cState.dead || self.cState.hazardDeath || self.cState.transitioning)
            {
                // ...Forza l'input del gioco a 0
                if (moveInputField != null) moveInputField.SetValue(self, 0f);
                
                // ...E IMPORTANTE: Cancella la memoria dell'AI!
                // Così quando il rinculo finisce, non ricomincia a camminare verso il boss.
                if (inputHandler != null) inputHandler.ResetState();
                
                return; 
            }

            if (inputHandler != null)
            {
                float aiInput = inputHandler.GetHorizontalInput();
                if (Mathf.Abs(aiInput) > 0.1f && moveInputField != null)
                {
                    moveInputField.SetValue(self, aiInput);
                    if (aiInput > 0) self.FaceRight(); else self.FaceLeft();
                }
            }
        }

        // --- HOOK LOGICA ---
        private void OnHeroUpdate()
        {
            var hero = HeroController.instance;
            if (hero == null) return;

            // Se sta prendendo danno, non fare nulla (il ResetState è già gestito sopra)
            if (hero.cState.recoiling || hero.cState.dead) return;

            float now = Time.time;

            // --- SALTO MIGLIORATO ---
            if (inputHandler.jumpRequested)
            {
                inputHandler.jumpRequested = false; // Consuma l'input

                // Logica salto più permissiva
                bool grounded = hero.cState.onGround;
                bool wallSliding = hero.cState.wallSliding;
                bool canJump = grounded || wallSliding;

                // Salta solo se NON stiamo già salendo (jumping)
                if (!hero.cState.jumping && canJump && (now - lastJumpTime > 0.25f))
                {
                    CallHeroMethod(hero, "Jump");
                    lastJumpTime = now;
                }
            }

            // --- ATTACCO ---
            if (inputHandler.attackRequested && (now - lastAttackTime > 0.3f))
            {
                CallHeroMethod(hero, "DoAttack");
                lastAttackTime = now;
                inputHandler.attackRequested = false;
            }

            // --- DASH ---
            if (inputHandler.dashRequested && (now - lastDashTime > 0.6f))
            {
                CallHeroMethod(hero, "HeroDash"); 
                lastDashTime = now;
                inputHandler.dashRequested = false;
            }
            
            // --- FOCUS ---
            if (inputHandler.focusRequested) { /* Placeholder */ }

            // --- DATI ---
            int currentHP = PlayerData.instance.health;
            if (lastHP == -1) lastHP = currentHP; 
            if (currentHP < lastHP) damageTakenSinceLastSend = true;
            lastHP = currentHP;

            updateCounter++;
            if (updateCounter % FRAME_SKIP == 0)
            {
                CaptureStateForAI(hero);
                damageTakenSinceLastSend = false;
            }
        }

        private void CaptureStateForAI(HeroController hero)
        {
            Vector3 pos = hero.transform.position;
            Vector2 vel = hero.GetComponent<Rigidbody2D>().velocity;
            int maxHp = PlayerData.instance.maxHealth;
            int soul = PlayerData.instance.MPCharge;

            List<GameStateCapture.Enemy> enemies = new List<GameStateCapture.Enemy>();
            var allEnemies = UnityEngine.Object.FindObjectsOfType<HealthManager>();
            foreach (var hm in allEnemies)
            {
                if (hm.hp <= 0 || hm.gameObject == hero.gameObject) continue;
                float dist = Vector2.Distance(pos, hm.transform.position);
                if (dist < 25f) 
                {
                    enemies.Add(new GameStateCapture.Enemy 
                    {
                        Type = hm.gameObject.name, X = hm.transform.position.x, Y = hm.transform.position.y, Distance = dist, HP = hm.hp
                    });
                }
            }

            stateCapture.UpdatePlayerState(pos, lastHP, maxHp, vel, soul, enemies, damageTakenSinceLastSend, hero.cState.onGround, hero.cState.touchingWall);
        }

        private void CallHeroMethod(HeroController hero, string methodName)
        {
            try {
                MethodInfo mi = typeof(HeroController).GetMethod(methodName, BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                if (mi != null) mi.Invoke(hero, null);
            } catch {}
        }
    }
}
