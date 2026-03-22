using System;
using Modding;
using InControl;
using UnityEngine;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Virtual Input Device per Hollow Knight - Basato su HKRL
    /// Usa bool state invece di DateTime expiration per maggiore stabilità
    /// </summary>
    public class SyntheticSoulInputDevice : InputDevice
    {
        // State flags
        private bool KeyUp = false;
        private bool KeyDown = false;
        private bool KeyLeft = false;
        private bool KeyRight = false;
        private bool KeyJump = false;
        private bool KeyAttack = false;
        private bool KeyDash = false;
        private bool KeyCast = false;

        public InputControl UpControl { get; private set; }
        public InputControl DownControl { get; private set; }
        public InputControl LeftControl { get; private set; }
        public InputControl RightControl { get; private set; }
        public InputControl JumpControl { get; private set; }
        public InputControl AttackControl { get; private set; }
        public InputControl DashControl { get; private set; }
        public InputControl CastControl { get; private set; }

        public SyntheticSoulInputDevice() : base("Synthetic Soul Virtual Controller")
        {
            UpControl = AddControl(InputControlType.DPadUp, "Up");
            DownControl = AddControl(InputControlType.DPadDown, "Down");
            LeftControl = AddControl(InputControlType.DPadLeft, "Left");
            RightControl = AddControl(InputControlType.DPadRight, "Right");
            JumpControl = AddControl(InputControlType.Action1, "Jump");
            CastControl = AddControl(InputControlType.Action2, "Cast");
            AttackControl = AddControl(InputControlType.Action3, "Attack");
            DashControl = AddControl(InputControlType.RightTrigger, "Dash");

            DesktopLogger.Log("✓ SyntheticSoulInputDevice created (HKRL-style)");
        }

        public override void Update(ulong updateTick, float deltaTime)
        {
            UpdateWithState(InputControlType.DPadUp, KeyUp, updateTick, deltaTime);
            UpdateWithState(InputControlType.DPadDown, KeyDown, updateTick, deltaTime);
            UpdateWithState(InputControlType.DPadLeft, KeyLeft, updateTick, deltaTime);
            UpdateWithState(InputControlType.DPadRight, KeyRight, updateTick, deltaTime);
            UpdateWithState(InputControlType.Action1, KeyJump, updateTick, deltaTime);
            UpdateWithState(InputControlType.Action2, KeyCast, updateTick, deltaTime);
            UpdateWithState(InputControlType.Action3, KeyAttack, updateTick, deltaTime);
            UpdateWithValue(InputControlType.RightTrigger, KeyDash ? 1 : 0, updateTick, deltaTime);
        }

        // ============ HELPER METHODS (come in HKRL) ============

        private static bool CanDash()
        {
            if (HeroController.instance == null) return false;
            return ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanDash");
        }

        private static bool CanAttack()
        {
            if (HeroController.instance == null) return false;
            return ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanAttack");
        }

        private static bool CanJump()
        {
            if (HeroController.instance == null) return false;
            return ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanJump");
        }

        private static bool CanDoubleJump()
        {
            if (HeroController.instance == null) return false;
            return ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanDoubleJump");
        }

        private static bool CanCast()
        {
            if (HeroController.instance == null) return false;
            return ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanCast");
        }

        private static bool CanWallJump()
        {
            if (HeroController.instance == null) return false;
            return ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanWallJump");
        }

        // ============ PUBLIC ACTION METHODS ============

        public void Reset()
        {
            KeyUp = false;
            KeyDown = false;
            KeyLeft = false;
            KeyRight = false;
            KeyJump = false;
            KeyAttack = false;
            KeyDash = false;
            KeyCast = false;
        }

        public void Left()
        {
            KeyLeft = true;
            KeyRight = false;
        }

        public void Right()
        {
            KeyRight = true;
            KeyLeft = false;
        }

        public void Up()
        {
            KeyUp = true;
            // NON resettare KeyDown - necessario per spell direzionali
        }

        public void Down()
        {
            KeyDown = true;
            // NON resettare KeyUp - necessario per spell direzionali
        }

        public void Jump()
        {
            if (!CanJump() && !CanDoubleJump() && !CanWallJump())
                return;

            KeyJump = true;
            KeyDash = false;
        }

        public void Attack()
        {
            if (!CanAttack())
                return;

            if (KeyLeft && HeroController.instance != null)
                HeroController.instance.FaceLeft();
            if (KeyRight && HeroController.instance != null)
                HeroController.instance.FaceRight();

            KeyAttack = true;
            KeyCast = false;
        }

        public void Dash()
        {
            if (!CanDash())
                return;

            if (KeyLeft && HeroController.instance != null)
                HeroController.instance.FaceLeft();
            else if (KeyRight && HeroController.instance != null)
                HeroController.instance.FaceRight();

            KeyDash = true;
            KeyJump = false;
        }

        public void Cast()
        {
            if (!CanCast())
                return;

            if (KeyLeft && HeroController.instance != null)
                HeroController.instance.FaceLeft();
            if (KeyRight && HeroController.instance != null)
                HeroController.instance.FaceRight();

            KeyCast = true;
            KeyAttack = false;
        }

        // Stop methods per rilasciare input
        public void StopLR()
        {
            KeyLeft = false;
            KeyRight = false;
        }

        public void StopUD()
        {
            KeyUp = false;
            KeyDown = false;
        }

        public void StopJump()
        {
            KeyJump = false;
        }

        public void StopDash()
        {
            KeyDash = false;
        }

        public void StopAttack()
        {
            KeyAttack = false;
        }

        public void StopCast()
        {
            KeyCast = false;
        }
    }

    public class ActionExecutor
    {
        private HeroController hero;
        private SyntheticSoulInputDevice device;
        private bool deviceAttached = false;
        private bool actionsBindingComplete = false;

        private object inputHandler;
        private object heroActions;

        // Timer per auto-release degli input
        private float jumpReleaseTime = 0f;
        private float attackReleaseTime = 0f;
        private float dashReleaseTime = 0f;
        private float castReleaseTime = 0f;
        private float moveReleaseTime = 0f;
        private float lookReleaseTime = 0f;

        private const float TAP_DURATION = 0.05f;
        private const float MOVEMENT_DURATION = 0.15f;

        public ActionExecutor()
        {
            DesktopLogger.Log("═══ ACTIONEXECUTOR V47 (HKRL-INSPIRED) ═══");
            InitializeDevice();
        }

        private void InitializeDevice()
        {
            try
            {
                device = new SyntheticSoulInputDevice();
                if (InputManager.Devices == null)
                    return;

                InputManager.AttachDevice(device);
                deviceAttached = true;
                DesktopLogger.Log("✓ Device attached to InputManager");
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"Device init failed: {e.Message}");
            }
        }

        private void BindDeviceToHeroActions()
        {
            if (actionsBindingComplete || hero == null)
                return;

            try
            {
                var inputHandlerField = typeof(HeroController).GetField("inputHandler",
                    System.Reflection.BindingFlags.NonPublic |
                    System.Reflection.BindingFlags.Instance |
                    System.Reflection.BindingFlags.Public);

                if (inputHandlerField == null)
                    return;

                inputHandler = inputHandlerField.GetValue(hero);
                var heroActionsField = inputHandler.GetType().GetField("inputActions",
                    System.Reflection.BindingFlags.NonPublic |
                    System.Reflection.BindingFlags.Instance |
                    System.Reflection.BindingFlags.Public);

                if (heroActionsField == null)
                    return;

                heroActions = heroActionsField.GetValue(inputHandler);
                Type haType = heroActions.GetType();

                BindAction(haType, "left", device.LeftControl);
                BindAction(haType, "right", device.RightControl);
                BindAction(haType, "up", device.UpControl);
                BindAction(haType, "down", device.DownControl);
                BindAction(haType, "jump", device.JumpControl);
                BindAction(haType, "attack", device.AttackControl);
                BindAction(haType, "dash", device.DashControl);
                BindAction(haType, "cast", device.CastControl);

                actionsBindingComplete = true;
                DesktopLogger.Log("✓ Actions bound to hero");
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
                var field = type.GetField(name,
                    System.Reflection.BindingFlags.NonPublic |
                    System.Reflection.BindingFlags.Instance |
                    System.Reflection.BindingFlags.Public);

                if (field != null)
                {
                    var action = field.GetValue(heroActions) as PlayerAction;
                    if (action != null && control != null)
                    {
                        action.AddBinding(new DeviceBindingSource(control.Target));
                    }
                }
            }
            catch { }
        }

        public void ExecuteAction(string action, bool force = false)
        {
            if (hero == null)
                hero = HeroController.instance;

            if (hero == null || hero.cState.dead)
                return;

            if (!force && !hero.acceptingInput)
            {
                device?.Reset();
                return;
            }

            try
            {
                string cmd = action.ToUpper().Trim();
                float timeNow = Time.time;

                switch (cmd)
                {
                    case "MOVE_LEFT":
                        device?.Left();
                        moveReleaseTime = timeNow + MOVEMENT_DURATION;
                        break;

                    case "MOVE_RIGHT":
                        device?.Right();
                        moveReleaseTime = timeNow + MOVEMENT_DURATION;
                        break;

                    case "UP":
                        device?.Up();
                        lookReleaseTime = timeNow + MOVEMENT_DURATION;
                        break;

                    case "DOWN":
                        device?.Down();
                        lookReleaseTime = timeNow + MOVEMENT_DURATION;
                        break;

                    case "JUMP":
                        device?.Jump();
                        jumpReleaseTime = timeNow + 0.15f;
                        break;

                    case "ATTACK":
                        device?.Attack();
                        attackReleaseTime = timeNow + TAP_DURATION;
                        break;

                    case "DASH":
                        device?.Dash();
                        dashReleaseTime = timeNow + TAP_DURATION;
                        break;

                    case "SPELL":
                    case "CAST":
                        device?.Cast();
                        castReleaseTime = timeNow + TAP_DURATION;
                        break;

                    case "IDLE":
                        device?.Reset();
                        ResetAllTimers();
                        break;

                    // ============ COMBO ACTIONS ============
                    case "JUMP_ATTACK":
                        // Jump + Attack simultanei (essenziale vs Mantis Lords)
                        device?.Jump();
                        device?.Attack();
                        jumpReleaseTime = timeNow + 0.15f;
                        attackReleaseTime = timeNow + TAP_DURATION;
                        break;

                    case "DASH_ATTACK":
                        // Dash + Attack (per closing distance e colpire)
                        device?.Dash();
                        device?.Attack();
                        dashReleaseTime = timeNow + TAP_DURATION;
                        attackReleaseTime = timeNow + TAP_DURATION + 0.05f; // Leggero delay
                        break;

                    case "DASH_LEFT":
                        // Dash verso sinistra
                        device?.Left();
                        device?.Dash();
                        moveReleaseTime = timeNow + TAP_DURATION;
                        dashReleaseTime = timeNow + TAP_DURATION;
                        break;

                    case "DASH_RIGHT":
                        // Dash verso destra
                        device?.Right();
                        device?.Dash();
                        moveReleaseTime = timeNow + TAP_DURATION;
                        dashReleaseTime = timeNow + TAP_DURATION;
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

            if (hero == null)
                return;

            if (!hero.acceptingInput || hero.cState.dead || hero.cState.transitioning)
            {
                device?.Reset();
                ResetAllTimers();
                return;
            }

            if (!actionsBindingComplete && deviceAttached)
                BindDeviceToHeroActions();

            // Auto-release degli input scaduti
            float timeNow = Time.time;

            if (timeNow > moveReleaseTime)
                device?.StopLR();

            if (timeNow > lookReleaseTime)
                device?.StopUD();

            if (timeNow > jumpReleaseTime)
                device?.StopJump();

            if (timeNow > attackReleaseTime)
                device?.StopAttack();

            if (timeNow > dashReleaseTime)
                device?.StopDash();

            if (timeNow > castReleaseTime)
                device?.StopCast();
        }

        private void ResetAllTimers()
        {
            jumpReleaseTime = 0f;
            attackReleaseTime = 0f;
            dashReleaseTime = 0f;
            castReleaseTime = 0f;
            moveReleaseTime = 0f;
            lookReleaseTime = 0f;
        }

        public void DestroyDevice()
        {
            DesktopLogger.Log("ActionExecutor: Destroying device...");

            if (device != null)
            {
                device.Reset();

                if (deviceAttached)
                {
                    InputManager.DetachDevice(device);
                    deviceAttached = false;
                }

                device = null;
            }

            actionsBindingComplete = false;
            DesktopLogger.Log("✓ Device destroyed");
        }
    }
}