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
        
        // --- REFLECTION FIELDS ---
        private FieldInfo moveInputField;
        private FieldInfo verticalInputField; 
        private FieldInfo attackInputField; 
        private FieldInfo castInputField;     
        private FieldInfo quickCastInputField;
        private FieldInfo dashInputField;
        private FieldInfo spellControlField; 
        private MonoBehaviour spellCtrlFSM;
        
        // Timers & Logic
        private float lastJumpTime = 0f;
        private float lastAttackTime = 0f;
        private float lastDashTime = 0f;
        private float lastCastTime = 0f;
        
        private float upStickyTimer = 0f;
        private float downStickyTimer = 0f;
        private const float STICKY_DURATION = 0.35f; 
        
        private int updateCounter = 0;      
        private const int FRAME_SKIP = 6; 
        private int lastHP = -1;
        private bool damageTakenSinceLastSend = false; 

        public override string GetVersion() => "98.5.0-GROUNDED-ONLY";

        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            Log("Inizializzazione Synthetic Soul (V98.5 - Grounded Only)...");
            try
            {
                stateCapture = new GameStateCapture();
                inputHandler = new InputHandler();
                socketServer = new SocketServer(8888, this); 

                socketServer.ActionCallback = (actionString) => 
                {
                    if (string.IsNullOrEmpty(actionString)) return;
                    string[] commands = actionString.Split(new char[] { '\n', '\r', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (string cmd in commands)
                    {
                        string cleanAction = cmd.Trim().ToUpperInvariant();
                        try 
                        {
                            if (cleanAction == "UP" || cleanAction == "ATTACK_UP") upStickyTimer = Time.time + STICKY_DURATION;
                            if (cleanAction == "DOWN" || cleanAction == "ATTACK_DOWN") downStickyTimer = Time.time + STICKY_DURATION;
                            var actionEnum = (InputHandler.AIAction)Enum.Parse(typeof(InputHandler.AIAction), cleanAction, true);
                            inputHandler.ExecuteAction(actionEnum);
                        }
                        catch {}
                    }
                };

                var flags = BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public;
                
                moveInputField = typeof(HeroController).GetField("move_input", flags) ?? typeof(HeroController).GetField("moveInput", flags);
                verticalInputField = typeof(HeroController).GetField("vertical_input", flags) ?? typeof(HeroController).GetField("verticalInput", flags);
                attackInputField = typeof(HeroController).GetField("attack_input", flags) ?? typeof(HeroController).GetField("attackInput", flags);
                quickCastInputField = typeof(HeroController).GetField("quick_cast_input", flags) ?? typeof(HeroController).GetField("quickCastInput", flags);
                spellControlField = typeof(HeroController).GetField("spellControl", flags);

                Thread serverThread = new Thread(() => socketServer.Start(stateCapture));
                serverThread.IsBackground = true;
                serverThread.Start();

                On.HeroController.LookForInput += OnLookForInput;
                ModHooks.HeroUpdateHook += OnHeroUpdate;
                On.HeroController.JumpReleased += OnJumpReleased; // Necessario per il salto variabile
                On.HeroController.Start += OnHeroStart;

                Log("Synthetic Soul Ready - V98.5 STABLE");
            }
            catch (Exception e) { Log("Init Error: " + e.Message); }
        }

        private void OnHeroStart(On.HeroController.orig_Start orig, HeroController self)
        {
            orig(self);
            LocateFSMs(self);
        }

        private void LocateFSMs(HeroController hero)
        {
            try { if (spellCtrlFSM == null && spellControlField != null) spellCtrlFSM = spellControlField.GetValue(hero) as MonoBehaviour; } catch {}
        }

        private void OnJumpReleased(On.HeroController.orig_JumpReleased orig, HeroController self)
        {
            // Se l'AI tiene premuto SALTO, ignoriamo il rilascio naturale del tasto (se serve)
            // MA se il timer AI è scaduto, lasciamo che il gioco processi il rilascio.
            bool aiJumping = false;
            if (inputHandler != null) aiJumping = inputHandler.jumpHeld;
            
            if (aiJumping) return; // L'AI sta ancora "premendo", non interrompere il salto.
            
            orig(self);
        }

        private void OnLookForInput(On.HeroController.orig_LookForInput orig, HeroController self)
        {
            orig(self);
            
            // 1. SICUREZZA: Se morto, in transizione o PRENDE DANNO (Recoil), STOP AI.
            if (self.cState.recoiling || self.cState.dead || self.cState.transitioning) return;
            
            if (inputHandler != null)
            {
                if (spellCtrlFSM == null) LocateFSMs(self);

                // === MOVIMENTO IBRIDO ===
                float aiInputX = inputHandler.GetHorizontalInput();
                if (Mathf.Abs(aiInputX) > 0.1f && moveInputField != null)
                {
                    moveInputField.SetValue(self, aiInputX);
                }

                // === SGUARDO ===
                if (verticalInputField != null)
                {
                    bool stickyUp = Time.time < upStickyTimer;
                    bool stickyDown = Time.time < downStickyTimer;

                    if (stickyUp || inputHandler.isLookingUp) 
                    {
                        verticalInputField.SetValue(self, 1.0f);
                        self.cState.lookingUp = true; self.cState.lookingDown = false;
                    }
                    else if (stickyDown || inputHandler.isLookingDown) 
                    {
                        verticalInputField.SetValue(self, -1.0f);
                        self.cState.lookingDown = true; self.cState.lookingUp = false;
                    }
                }

                // Attacchi e Cast simulati
                if (inputHandler.attackHeld && attackInputField != null) attackInputField.SetValue(self, true);
                if (inputHandler.castHeld && quickCastInputField != null) quickCastInputField.SetValue(self, true);
            }
        }

        private void OnHeroUpdate()
        {
            var hero = HeroController.instance;
            if (hero == null || hero.cState.dead) return;
            
            // CRUCIALE: Se prende danno, ferma tutto.
            if (hero.cState.recoiling) return;

            float now = Time.time;
            bool isCasting = inputHandler.castHeld;  
            bool stickyUp = Time.time < upStickyTimer;
            bool stickyDown = Time.time < downStickyTimer;

            // --- SPELL ---
            if (isCasting && (now - lastCastTime > 0.4f))
            {
                if (!stickyUp && !stickyDown && verticalInputField != null) verticalInputField.SetValue(hero, 0f);
                if (spellCtrlFSM != null) { SendFSMEvent(spellCtrlFSM, "QUICK CAST"); lastCastTime = now; }
            }
            
            // --- ATTACK ---
            if (inputHandler.attackHeld && (now - lastAttackTime > 0.35f))
            {
                if (stickyUp || inputHandler.isLookingUp)
                {
                    hero.cState.lookingUp = true; hero.cState.lookingDown = false;
                    if (verticalInputField != null) verticalInputField.SetValue(hero, 1.0f);
                }
                else if (stickyDown || inputHandler.isLookingDown)
                {
                    hero.cState.lookingDown = true; hero.cState.lookingUp = false;
                    if (verticalInputField != null) verticalInputField.SetValue(hero, -1.0f);
                }
                CallHeroMethod(hero, "DoAttack");
                lastAttackTime = now;
            }

            // --- JUMP (VECCHIO METODO + CONTROLLI) ---
            if (inputHandler.jumpHeld)
            {
                // CONTROLLO 1: Posso saltare solo se sono a terra o scivolo su un muro.
                bool canJump = hero.cState.onGround || hero.cState.wallSliding;
                
                // CONTROLLO 2: Non devo stare già saltando (evita spam).
                bool alreadyJumping = hero.cState.jumping;

                if (canJump && !alreadyJumping)
                {
                    // Cooldown minimo di 0.2s tra i salti
                    if (now - lastJumpTime > 0.2f) 
                    { 
                        CallHeroMethod(hero, "Jump"); 
                        lastJumpTime = now; 
                    }
                }
            }

            // --- DASH ---
            if (inputHandler.dashHeld && (now - lastDashTime > 0.6f)) { CallHeroMethod(hero, "HeroDash"); lastDashTime = now; }

            // --- DATA CAPTURE ---
            int currentHP = PlayerData.instance.health;
            if (lastHP == -1) lastHP = currentHP; 
            if (currentHP < lastHP) damageTakenSinceLastSend = true;
            lastHP = currentHP;
            
            updateCounter++;
            if (updateCounter % FRAME_SKIP == 0) { CaptureStateForAI(hero); damageTakenSinceLastSend = false; }
        }

        private void SendFSMEvent(object fsmObject, string eventName)
        {
            if (fsmObject == null) return;
            try { MethodInfo method = fsmObject.GetType().GetMethod("SendEvent", new Type[] { typeof(string) }); if (method != null) method.Invoke(fsmObject, new object[] { eventName }); } catch {}
        }
        private void CallHeroMethod(HeroController hero, string methodName) { try { MethodInfo mi = typeof(HeroController).GetMethod(methodName, BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public); if (mi != null) mi.Invoke(hero, null); } catch {} }
        
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
                if (dist < 50f) enemies.Add(new GameStateCapture.Enemy { Type = hm.gameObject.name, X = hm.transform.position.x, Y = hm.transform.position.y, Distance = dist, HP = hm.hp });
            } 
            stateCapture.UpdatePlayerState(pos, lastHP, maxHp, vel, soul, enemies, damageTakenSinceLastSend, hero.cState.onGround, hero.cState.touchingWall);
        }
    }
}