using System;
using System.Linq;
using UnityEngine;
using System.Reflection;
using InControl;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Complete Virtual Input Device - simulates a full controller
    /// </summary>
    public class SyntheticSoulInputDevice : InputDevice
    {
        // Directional controls
        public InputControl LeftControl { get; private set; }
        public InputControl RightControl { get; private set; }
        public InputControl UpControl { get; private set; }
        public InputControl DownControl { get; private set; }
        
        // Action controls
        public InputControl JumpControl { get; private set; }
        public InputControl AttackControl { get; private set; }
        public InputControl DashControl { get; private set; }
        public InputControl CastControl { get; private set; }

        public SyntheticSoulInputDevice() : base("Synthetic Soul Virtual Controller")
        {
            DesktopLogger.Log("Creating SyntheticSoulInputDevice (FULL CONTROLLER)...");
            
            // Directional inputs
            LeftControl = AddControl(InputControlType.LeftStickLeft, "Left");
            RightControl = AddControl(InputControlType.LeftStickRight, "Right");
            UpControl = AddControl(InputControlType.LeftStickUp, "Up");
            DownControl = AddControl(InputControlType.LeftStickDown, "Down");
            
            // Action buttons
            JumpControl = AddControl(InputControlType.Action1, "Jump");
            AttackControl = AddControl(InputControlType.Action2, "Attack");
            DashControl = AddControl(InputControlType.Action3, "Dash");
            CastControl = AddControl(InputControlType.Action4, "Cast");
            
            DesktopLogger.Log("✓ SyntheticSoulInputDevice created with 8 controls");
        }
    }

    public class ActionExecutor
    {
        private HeroController hero;
        
        // Virtual device
        private SyntheticSoulInputDevice virtualDevice;
        private bool deviceAttached = false;
        private bool actionsBindingComplete = false;
        
        // Input state - what buttons should be "held" this frame
        private bool wantLeft = false;
        private bool wantRight = false;
        private bool wantUp = false;
        private bool wantDown = false;
        private bool wantJump = false;
        private bool wantAttack = false;
        private bool wantDash = false;
        private bool wantCast = false;
        
        // Action expiration times (for sustained presses)
        private DateTime leftExpiration = DateTime.MinValue;
        private DateTime rightExpiration = DateTime.MinValue;
        private DateTime upExpiration = DateTime.MinValue;
        private DateTime downExpiration = DateTime.MinValue;
        private DateTime jumpExpiration = DateTime.MinValue;
        private DateTime attackExpiration = DateTime.MinValue;
        private DateTime dashExpiration = DateTime.MinValue;
        private DateTime castExpiration = DateTime.MinValue;
        
        // Durations
        private const double MOVEMENT_DURATION = 0.5;  // Hold directional keys
        private const double ACTION_DURATION = 0.15;    // Tap action buttons
        private const double DIRECTIONAL_SETUP_DURATION = 0.1;  // Hold direction before action
        
        // Cooldown tracking
        private float lastJumpTime = 0f;
        private float lastAttackTime = 0f;
        private float lastDashTime = 0f;
        private float lastCastTime = 0f;
        
        // Reference for binding
        private object inputHandler;
        private object heroActions;

        public ActionExecutor()
        {
            DesktopLogger.Log("\n" + new string('=', 60));
            DesktopLogger.Log("=== ACTIONEXECUTOR - FULL VIRTUAL DEVICE MODE ===");
            DesktopLogger.Log("Virtual Device: ALL inputs (L/R/U/D + Jump/Attack/Dash/Cast)");
            DesktopLogger.Log("NO Reflection - Pure InControl simulation");
            DesktopLogger.Log(new string('=', 60) + "\n");
            
            InitializeVirtualDevice();
            
            DesktopLogger.Log("\n" + new string('=', 60));
            DesktopLogger.Log("=== INITIALIZATION COMPLETE ===");
            DesktopLogger.Log(new string('=', 60) + "\n");
            Modding.Logger.Log("[SyntheticSoul] ActionExecutor initialized - FULL VIRTUAL DEVICE");
        }

        private void InitializeVirtualDevice()
        {
            try
            {
                DesktopLogger.Log("\n--- CREATING FULL VIRTUAL INPUT DEVICE ---\n");
                virtualDevice = new SyntheticSoulInputDevice();
                
                if (InputManager.Devices == null)
                {
                    DesktopLogger.LogError("InputManager.Devices is null!");
                    return;
                }
                
                InputManager.AttachDevice(virtualDevice);
                deviceAttached = true;
                
                DesktopLogger.Log($"✓ Virtual device attached to InputManager");
                DesktopLogger.Log($"  Total devices: {InputManager.Devices.Count}");
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Failed to initialize virtual device: {e.Message}\n{e.StackTrace}");
            }
        }

        private void BindDeviceToHeroActions()
        {
            if (actionsBindingComplete) return;
            if (hero == null) return;
            
            try
            {
                DesktopLogger.Log("\n--- BINDING DEVICE TO HEROACTIONS (ALL CONTROLS) ---\n");
                
                // Get InputHandler
                FieldInfo inputHandlerField = typeof(HeroController).GetField("inputHandler",
                    BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                
                if (inputHandlerField == null)
                {
                    DesktopLogger.LogError("InputHandler field not found!");
                    return;
                }
                
                inputHandler = inputHandlerField.GetValue(hero);
                if (inputHandler == null)
                {
                    DesktopLogger.LogError("InputHandler instance is null!");
                    return;
                }
                
                // Get HeroActions
                Type inputHandlerType = inputHandler.GetType();
                FieldInfo heroActionsField = inputHandlerType.GetField("inputActions",
                    BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                
                if (heroActionsField == null)
                {
                    DesktopLogger.LogError("inputActions field not found!");
                    return;
                }
                
                heroActions = heroActionsField.GetValue(inputHandler);
                if (heroActions == null)
                {
                    DesktopLogger.LogError("HeroActions instance is null!");
                    return;
                }
                
                DesktopLogger.Log($"✓ Got HeroActions instance");
                
                // Bind ALL controls
                Type heroActionsType = heroActions.GetType();
                BindAction(heroActionsType, "left", virtualDevice.LeftControl);
                BindAction(heroActionsType, "right", virtualDevice.RightControl);
                BindAction(heroActionsType, "up", virtualDevice.UpControl);
                BindAction(heroActionsType, "down", virtualDevice.DownControl);
                BindAction(heroActionsType, "jump", virtualDevice.JumpControl);
                BindAction(heroActionsType, "attack", virtualDevice.AttackControl);
                BindAction(heroActionsType, "evade", virtualDevice.DashControl);  // Dash is "evade"
                BindAction(heroActionsType, "cast", virtualDevice.CastControl);
                
                actionsBindingComplete = true;
                DesktopLogger.Log("\n✓ ALL CONTROLS BOUND - Full virtual controller active!");
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Failed to bind device: {e.Message}\n{e.StackTrace}");
            }
        }

        private void BindAction(Type heroActionsType, string actionName, InputControl control)
        {
            try
            {
                FieldInfo actionField = heroActionsType.GetField(actionName,
                    BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                
                if (actionField == null)
                {
                    DesktopLogger.LogWarning($"  Action '{actionName}' field not found");
                    return;
                }
                
                object action = actionField.GetValue(heroActions);
                if (action == null)
                {
                    DesktopLogger.LogWarning($"  Action '{actionName}' instance is null");
                    return;
                }
                
                Type actionType = action.GetType();
                MethodInfo addBindingMethod = actionType.GetMethod("AddDefaultBinding",
                    new Type[] { typeof(InputControl) });
                
                if (addBindingMethod == null)
                {
                    DesktopLogger.LogWarning($"  AddDefaultBinding method not found for '{actionName}'");
                    return;
                }
                
                addBindingMethod.Invoke(action, new object[] { control });
                DesktopLogger.Log($"  ✓ Bound {actionName}");
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Failed to bind action '{actionName}': {e.Message}");
            }
        }

        public void ExecuteAction(string action)
        {
            if (hero == null)
                hero = HeroController.instance;
            
            if (hero == null || hero.cState.dead || hero.cState.recoiling)
                return;
            
            try
            {
                string actionCmd = action.ToUpper().Trim();
                DateTime now = DateTime.Now;
                float timeNow = Time.time;
                
                switch (actionCmd)
                {
                    case "MOVE_LEFT":
                        leftExpiration = now.AddSeconds(MOVEMENT_DURATION);
                        rightExpiration = DateTime.MinValue;  // Cancel opposite
                        DesktopLogger.Log("MOVE_LEFT command");
                        break;
                    
                    case "MOVE_RIGHT":
                        rightExpiration = now.AddSeconds(MOVEMENT_DURATION);
                        leftExpiration = DateTime.MinValue;  // Cancel opposite
                        DesktopLogger.Log("MOVE_RIGHT command");
                        break;
                    
                    case "JUMP":
                        if (timeNow - lastJumpTime > 0.2f)
                        {
                            jumpExpiration = now.AddSeconds(ACTION_DURATION);
                            lastJumpTime = timeNow;
                            DesktopLogger.Log("JUMP command");
                        }
                        break;
                    
                    case "ATTACK":
                        // Neutral attack
                        if (timeNow - lastAttackTime > 0.35f)
                        {
                            attackExpiration = now.AddSeconds(ACTION_DURATION);
                            lastAttackTime = timeNow;
                            DesktopLogger.Log("ATTACK (neutral) command");
                        }
                        break;
                    
                    case "ATTACK_UP":
                        if (timeNow - lastAttackTime > 0.35f)
                        {
                            // Hold UP, then press ATTACK
                            upExpiration = now.AddSeconds(DIRECTIONAL_SETUP_DURATION + ACTION_DURATION);
                            attackExpiration = now.AddSeconds(ACTION_DURATION);
                            lastAttackTime = timeNow;
                            DesktopLogger.Log("ATTACK_UP command (Up + Attack)");
                        }
                        break;
                    
                    case "ATTACK_DOWN":
                        if (timeNow - lastAttackTime > 0.35f)
                        {
                            // Hold DOWN, then press ATTACK
                            downExpiration = now.AddSeconds(DIRECTIONAL_SETUP_DURATION + ACTION_DURATION);
                            attackExpiration = now.AddSeconds(ACTION_DURATION);
                            lastAttackTime = timeNow;
                            DesktopLogger.Log("ATTACK_DOWN command (Down + Attack)");
                        }
                        break;
                    
                    case "DASH":
                        if (timeNow - lastDashTime > 0.6f)
                        {
                            dashExpiration = now.AddSeconds(ACTION_DURATION);
                            lastDashTime = timeNow;
                            DesktopLogger.Log("DASH command");
                        }
                        break;
                    
                    case "DASH_LEFT":
                        if (timeNow - lastDashTime > 0.6f)
                        {
                            leftExpiration = now.AddSeconds(DIRECTIONAL_SETUP_DURATION + ACTION_DURATION);
                            dashExpiration = now.AddSeconds(ACTION_DURATION);
                            rightExpiration = DateTime.MinValue;
                            lastDashTime = timeNow;
                            DesktopLogger.Log("DASH_LEFT command (Left + Dash)");
                        }
                        break;
                    
                    case "DASH_RIGHT":
                        if (timeNow - lastDashTime > 0.6f)
                        {
                            rightExpiration = now.AddSeconds(DIRECTIONAL_SETUP_DURATION + ACTION_DURATION);
                            dashExpiration = now.AddSeconds(ACTION_DURATION);
                            leftExpiration = DateTime.MinValue;
                            lastDashTime = timeNow;
                            DesktopLogger.Log("DASH_RIGHT command (Right + Dash)");
                        }
                        break;
                    
                    case "SPELL_UP":
                        if (PlayerData.instance != null && PlayerData.instance.MPCharge >= 33 
                            && timeNow - lastCastTime > 0.25f)
                        {
                            // Vengeful Spirit up (Shriek)
                            upExpiration = now.AddSeconds(DIRECTIONAL_SETUP_DURATION + ACTION_DURATION);
                            castExpiration = now.AddSeconds(ACTION_DURATION);
                            lastCastTime = timeNow;
                            DesktopLogger.Log("SPELL_UP command (Up + Cast = Shriek)");
                        }
                        break;
                    
                    case "SPELL_DOWN":
                        if (PlayerData.instance != null && PlayerData.instance.MPCharge >= 33 
                            && timeNow - lastCastTime > 0.25f)
                        {
                            // Vengeful Spirit down (Dive)
                            downExpiration = now.AddSeconds(DIRECTIONAL_SETUP_DURATION + ACTION_DURATION);
                            castExpiration = now.AddSeconds(ACTION_DURATION);
                            lastCastTime = timeNow;
                            DesktopLogger.Log("SPELL_DOWN command (Down + Cast = Dive)");
                        }
                        break;
                    
                    case "SPELL_SIDE":
                    case "SPELL":
                        if (PlayerData.instance != null && PlayerData.instance.MPCharge >= 33 
                            && timeNow - lastCastTime > 0.25f)
                        {
                            // Neutral spell (Vengeful Spirit / Shade Soul)
                            castExpiration = now.AddSeconds(ACTION_DURATION);
                            lastCastTime = timeNow;
                            DesktopLogger.Log("SPELL_SIDE command (Cast = Fireball)");
                        }
                        break;
                    
                    case "IDLE":
                        // Do nothing
                        break;
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"ExecuteAction error: {e.Message}");
            }
        }

        public void Update()
        {
            if (hero == null)
                hero = HeroController.instance;
            
            if (hero == null || hero.cState.transitioning)
                return;
            
            // Bind device to actions on first update
            if (!actionsBindingComplete && deviceAttached)
            {
                BindDeviceToHeroActions();
            }
            
            DateTime now = DateTime.Now;
            float deltaTime = Time.deltaTime;
            
            // ========================================
            // STEP 1: DETERMINE WHICH BUTTONS TO PRESS
            // ========================================
            
            // Directional inputs
            wantLeft = (now < leftExpiration);
            wantRight = (now < rightExpiration);
            wantUp = (now < upExpiration);
            wantDown = (now < downExpiration);
            
            // Action inputs
            wantJump = (now < jumpExpiration);
            wantAttack = (now < attackExpiration);
            wantDash = (now < dashExpiration);
            wantCast = (now < castExpiration);
            
            // ========================================
            // STEP 2: UPDATE VIRTUAL DEVICE STATE
            // ========================================
            
            if (deviceAttached && !hero.cState.recoiling && !hero.cState.dead)
            {
                UpdateVirtualDeviceState(deltaTime);
            }
        }

        private void UpdateVirtualDeviceState(float deltaTime)
        {
            if (virtualDevice == null) return;
            
            try
            {
                ulong currentTick = InputManager.CurrentTick;
                
                // Update all controls
                virtualDevice.LeftControl.UpdateWithState(wantLeft, currentTick, deltaTime);
                virtualDevice.RightControl.UpdateWithState(wantRight, currentTick, deltaTime);
                virtualDevice.UpControl.UpdateWithState(wantUp, currentTick, deltaTime);
                virtualDevice.DownControl.UpdateWithState(wantDown, currentTick, deltaTime);
                virtualDevice.JumpControl.UpdateWithState(wantJump, currentTick, deltaTime);
                virtualDevice.AttackControl.UpdateWithState(wantAttack, currentTick, deltaTime);
                virtualDevice.DashControl.UpdateWithState(wantDash, currentTick, deltaTime);
                virtualDevice.CastControl.UpdateWithState(wantCast, currentTick, deltaTime);
                
                // Commit changes
                virtualDevice.Commit(currentTick, deltaTime);
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Failed to update device state: {e.Message}");
            }
        }

        public void DestroyDevice()
        {
            try
            {
                DesktopLogger.Log("Destroying virtual input device...");
                
                if (virtualDevice != null && deviceAttached)
                {
                    if (InputManager.Devices != null && InputManager.Devices.Contains(virtualDevice))
                    {
                        InputManager.DetachDevice(virtualDevice);
                        DesktopLogger.Log("✓ Virtual device detached from InputManager");
                    }
                    
                    virtualDevice = null;
                    deviceAttached = false;
                    actionsBindingComplete = false;
                }
                
                inputHandler = null;
                heroActions = null;
                
                DesktopLogger.Log("✓ ActionExecutor cleaned up");
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Error destroying device: {e.Message}");
            }
        }
    }
}