using System;
using System.Linq;
using UnityEngine;
using System.Reflection;
using InControl;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Virtual Controller con layout Xbox Standard:
    /// A=Jump, X=Attack, B=Cast/Focus, RT=Dash
    /// </summary>
    public class SyntheticSoulInputDevice : InputDevice
    {
        public InputControl LeftControl { get; private set; }
        public InputControl RightControl { get; private set; }
        public InputControl UpControl { get; private set; }
        public InputControl DownControl { get; private set; }
        public InputControl JumpControl { get; private set; }
        public InputControl AttackControl { get; private set; }
        public InputControl DashControl { get; private set; }
        public InputControl CastControl { get; private set; }

        public SyntheticSoulInputDevice() : base("Synthetic Soul Virtual Controller")
        {
            DesktopLogger.Log("Creating SyntheticSoulInputDevice (XBOX LAYOUT)...");

            // Stick Sinistro
            LeftControl = AddControl(InputControlType.LeftStickLeft, "Left");
            RightControl = AddControl(InputControlType.LeftStickRight, "Right");
            UpControl = AddControl(InputControlType.LeftStickUp, "Up");
            DownControl = AddControl(InputControlType.LeftStickDown, "Down");

            // Pulsanti
            JumpControl = AddControl(InputControlType.Action1, "Jump (A)");
            CastControl = AddControl(InputControlType.Action2, "Cast (B)");
            AttackControl = AddControl(InputControlType.Action3, "Attack (X)");

            // Grilletti
            DashControl = AddControl(InputControlType.RightTrigger, "Dash (RT)");

            DesktopLogger.Log("✓ Virtual Device Created: A=Jump, X=Attack, B=Cast, RT=Dash");
        }
    }

    public class ActionExecutor
    {
        private HeroController hero;
        private SyntheticSoulInputDevice virtualDevice;
        private bool deviceAttached = false;
        private bool actionsBindingComplete = false;

        // Input state
        private bool wantLeft, wantRight, wantUp, wantDown;
        private bool wantJump, wantAttack, wantDash, wantCast;

        // Expiration timers
        private DateTime leftExpiration, rightExpiration, upExpiration, downExpiration;
        private DateTime jumpExpiration, attackExpiration, dashExpiration, castExpiration;

        // Durations & Cooldowns
        private const double MOVEMENT_DURATION = 0.15;
        private const double TAP_DURATION = 0.05;

        private float lastJumpTime = 0f;
        private float lastAttackTime = 0f;
        private float lastDashTime = 0f;
        private float lastCastTime = 0f;
        private float lastLogTime = 0f;

        // Reflection vars
        private object inputHandler;
        private object heroActions;

        public ActionExecutor()
        {
            DesktopLogger.Log("═══ ACTIONEXECUTOR INITIALIZED (with FORCE support) ═══");
            InitializeVirtualDevice();
        }

        private void InitializeVirtualDevice()
        {
            try
            {
                virtualDevice = new SyntheticSoulInputDevice();
                if (InputManager.Devices == null) return;

                InputManager.AttachDevice(virtualDevice);
                deviceAttached = true;
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Device init failed: {e.Message}");
            }
        }

        private void BindDeviceToHeroActions()
        {
            if (actionsBindingComplete || hero == null) return;

            try
            {
                DesktopLogger.Log("--- BINDING CONTROLS ---");

                var inputHandlerField = typeof(HeroController).GetField("inputHandler",
                    BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);

                if (inputHandlerField == null) return;
                inputHandler = inputHandlerField.GetValue(hero);

                var heroActionsField = inputHandler.GetType().GetField("inputActions",
                    BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);

                if (heroActionsField == null) return;
                heroActions = heroActionsField.GetValue(inputHandler);

                Type haType = heroActions.GetType();

                BindAction(haType, "left", virtualDevice.LeftControl);
                BindAction(haType, "right", virtualDevice.RightControl);
                BindAction(haType, "up", virtualDevice.UpControl);
                BindAction(haType, "down", virtualDevice.DownControl);
                BindAction(haType, "jump", virtualDevice.JumpControl);
                BindAction(haType, "attack", virtualDevice.AttackControl);
                BindAction(haType, "dash", virtualDevice.DashControl);
                BindAction(haType, "cast", virtualDevice.CastControl);

                actionsBindingComplete = true;
                DesktopLogger.Log("✓ Controls Bound Successfully");
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Binding failed: {e.Message}");
            }
        }

        private void BindAction(Type type, string name, InputControl control)
        {
            try
            {
                FieldInfo field = type.GetField(name,
                    BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);

                if (field != null)
                {
                    var action = field.GetValue(heroActions) as PlayerAction;
                    if (action != null)
                    {
                        action.AddBinding(new DeviceBindingSource(control.Target));
                        DesktopLogger.Log($"✓ Bound '{name}' to {control.Target}");
                    }
                }
            }
            catch { }
        }

        /// <summary>
        /// Executes an action on the virtual controller.
        /// </summary>
        /// <param name="action">Action command (e.g., "JUMP", "ATTACK")</param>
        /// <param name="force">If true, bypass acceptingInput check (for wake-up)</param>
        public void ExecuteAction(string action, bool force = false)
        {
            if (hero == null)
                hero = HeroController.instance;

            if (hero == null || hero.cState.dead || hero.cState.recoiling)
                return;

            // ============ CRITICAL CHECK: ACCEPTING INPUT ============
            // UNLESS force=true (used for waking up sleeping hero)
            if (!force && !hero.acceptingInput)
            {
                // Log opzionale (throttled)
                if (Time.time - lastLogTime > 2.0f)
                {
                    DesktopLogger.Log("ActionExecutor: Blocking input - cutscene/spawn");
                    lastLogTime = Time.time;
                }
                ReleaseAllKeys();
                return;
            }

            // ============ FORCE MODE LOGGING ============
            if (force && !hero.acceptingInput)
            {
                DesktopLogger.Log($"[FORCE] Executing '{action}' despite acceptingInput=false (wake-up mode)");
            }

            try
            {
                string cmd = action.ToUpper().Trim();
                DateTime now = DateTime.Now;
                float timeNow = Time.time;

                switch (cmd)
                {
                    case "MOVE_LEFT":
                        leftExpiration = now.AddSeconds(MOVEMENT_DURATION);
                        rightExpiration = DateTime.MinValue;
                        break;

                    case "MOVE_RIGHT":
                        rightExpiration = now.AddSeconds(MOVEMENT_DURATION);
                        leftExpiration = DateTime.MinValue;
                        break;

                    case "UP":
                        upExpiration = now.AddSeconds(MOVEMENT_DURATION);
                        break;

                    case "DOWN":
                        downExpiration = now.AddSeconds(MOVEMENT_DURATION);
                        break;

                    case "JUMP":
                        if (timeNow - lastJumpTime > 0.2f)
                        {
                            jumpExpiration = now.AddSeconds(0.15);
                            lastJumpTime = timeNow;
                        }
                        break;

                    case "ATTACK":
                        if (timeNow - lastAttackTime > 0.35f)
                        {
                            attackExpiration = now.AddSeconds(TAP_DURATION);
                            lastAttackTime = timeNow;
                        }
                        break;

                    case "DASH":
                        if (timeNow - lastDashTime > 0.4f)
                        {
                            dashExpiration = now.AddSeconds(TAP_DURATION);
                            lastDashTime = timeNow;
                            DesktopLogger.Log("DASH (RT)");
                        }
                        break;

                    case "SPELL":
                    case "CAST":
                        if (PlayerData.instance.MPCharge >= 33 && timeNow - lastCastTime > 0.3f)
                        {
                            castExpiration = now.AddSeconds(TAP_DURATION);
                            lastCastTime = timeNow;
                            DesktopLogger.Log("SPELL (B)");
                        }
                        break;

                    case "DREAM_NAIL":
                        upExpiration = now.AddSeconds(0.3);
                        break;

                    case "IDLE":
                        ReleaseAllKeys();
                        break;
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Exec Error: {e.Message}");
            }
        }

        public void Update()
        {
            if (hero == null)
                hero = HeroController.instance;

            if (hero == null)
                return;

            if (!hero.acceptingInput || hero.cState.dead || hero.cState.transitioning)
            {
                ReleaseAllKeys();
                return;
            }

            if (!actionsBindingComplete && deviceAttached)
            {
                BindDeviceToHeroActions();
            }

            DateTime now = DateTime.Now;
            float dt = Time.deltaTime;

            wantLeft = now < leftExpiration;
            wantRight = now < rightExpiration;
            wantUp = now < upExpiration;
            wantDown = now < downExpiration;
            wantJump = now < jumpExpiration;
            wantAttack = now < attackExpiration;
            wantDash = now < dashExpiration;
            wantCast = now < castExpiration;

            if (deviceAttached && virtualDevice != null)
            {
                ulong tick = InputManager.CurrentTick;

                virtualDevice.LeftControl.UpdateWithState(wantLeft, tick, dt);
                virtualDevice.RightControl.UpdateWithState(wantRight, tick, dt);
                virtualDevice.UpControl.UpdateWithState(wantUp, tick, dt);
                virtualDevice.DownControl.UpdateWithState(wantDown, tick, dt);
                virtualDevice.JumpControl.UpdateWithState(wantJump, tick, dt);
                virtualDevice.AttackControl.UpdateWithState(wantAttack, tick, dt);
                virtualDevice.DashControl.UpdateWithState(wantDash, tick, dt);
                virtualDevice.CastControl.UpdateWithState(wantCast, tick, dt);

                virtualDevice.Commit(tick, dt);
            }
        }

        private void ReleaseAllKeys()
        {
            try
            {
                leftExpiration = DateTime.MinValue;
                rightExpiration = DateTime.MinValue;
                upExpiration = DateTime.MinValue;
                downExpiration = DateTime.MinValue;
                jumpExpiration = DateTime.MinValue;
                attackExpiration = DateTime.MinValue;
                dashExpiration = DateTime.MinValue;
                castExpiration = DateTime.MinValue;

                if (deviceAttached && virtualDevice != null)
                {
                    ulong tick = InputManager.CurrentTick;
                    float dt = Time.deltaTime;

                    virtualDevice.LeftControl.UpdateWithState(false, tick, dt);
                    virtualDevice.RightControl.UpdateWithState(false, tick, dt);
                    virtualDevice.UpControl.UpdateWithState(false, tick, dt);
                    virtualDevice.DownControl.UpdateWithState(false, tick, dt);
                    virtualDevice.JumpControl.UpdateWithState(false, tick, dt);
                    virtualDevice.AttackControl.UpdateWithState(false, tick, dt);
                    virtualDevice.DashControl.UpdateWithState(false, tick, dt);
                    virtualDevice.CastControl.UpdateWithState(false, tick, dt);

                    virtualDevice.Commit(tick, dt);
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"ActionExecutor: Error releasing keys: {e.Message}");
            }
        }

        public void DestroyDevice()
        {
            DesktopLogger.Log("ActionExecutor: Destroying virtual device...");

            ReleaseAllKeys();

            if (virtualDevice != null && deviceAttached)
            {
                InputManager.DetachDevice(virtualDevice);
                virtualDevice = null;
                deviceAttached = false;
                actionsBindingComplete = false;
            }

            DesktopLogger.Log("✓ ActionExecutor: Virtual device destroyed");
        }
    }
}