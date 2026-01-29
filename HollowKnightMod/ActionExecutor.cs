using System;
using System.Linq;
using UnityEngine;
using System.Reflection;
using InControl;
using Modding;

namespace SyntheticSoulMod
{
    // =================================================================================
    // FIX V43: ANIMATION STABILITY (LOOK UP/DOWN FIX)
    // Risolve il problema dello "spam" dell'animazione inchino/sguardo.
    // Forza lo stato cState.lookingUp/Down in LateUpdate per evitare che il gioco lo resetti.
    // =================================================================================
    [DefaultExecutionOrder(9999)]
    public class HeroPhysicsOverride : MonoBehaviour
    {
        private Rigidbody2D rb;
        private HeroController hero;
        private PropertyInfo runningProp;
        private tk2dSpriteAnimator animator;

        public float TargetVelocityX { get; set; } = 0f;
        public bool IsActive { get; set; } = false;

        public bool LockVelocity { get; set; } = false;
        public bool ForceIdleAnim { get; set; } = false;

        // V43: Nuovi flag per gestire esplicitamente lo sguardo
        public bool IsLookingUp { get; set; } = false;
        public bool IsLookingDown { get; set; } = false;

        void Awake()
        {
            rb = GetComponent<Rigidbody2D>();
            hero = GetComponent<HeroController>();
            animator = hero.GetComponent<tk2dSpriteAnimator>();

            if (hero != null && hero.cState != null)
            {
                runningProp = hero.cState.GetType().GetProperty("running",
                    BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic);
            }
        }

        void FixedUpdate()
        {
            if (IsActive && rb != null)
            {
                // V43: Se stiamo guardando su o giù, la velocità DEVE essere zero assoluto
                if (IsLookingUp || IsLookingDown)
                {
                    rb.velocity = Vector2.zero;
                    rb.angularVelocity = 0f;
                }
                else if (LockVelocity && hero.cState.onGround)
                {
                    rb.velocity = Vector2.zero;
                    rb.angularVelocity = 0f;
                }
                else
                {
                    rb.velocity = new Vector2(TargetVelocityX, rb.velocity.y);
                }
            }
        }

        void LateUpdate()
        {
            if (!IsActive || hero == null) return;

            // =========================================================
            // FIX ANIMAZIONI LOOK UP / LOOK DOWN
            // =========================================================
            if (IsLookingUp && hero.cState.onGround)
            {
                // Forza lo stato interno per evitare che il gioco lo resetti
                hero.cState.lookingUp = true;
                hero.cState.lookingDown = false;

                // Ferma input di movimento residui
                if (runningProp != null) try { runningProp.SetValue(hero.cState, false, null); } catch { }

                // Gestione sicura animazione
                PlayAnimSafe("LookUp");
                return; // Esce per evitare che ForceIdle sovrascriva
            }

            if (IsLookingDown && hero.cState.onGround)
            {
                hero.cState.lookingDown = true;
                hero.cState.lookingUp = false;

                if (runningProp != null) try { runningProp.SetValue(hero.cState, false, null); } catch { }

                PlayAnimSafe("LookDown");
                return; // Esce
            }

            // =========================================================
            // FIX IDLE
            // =========================================================
            if (ForceIdleAnim)
            {
                if (rb != null) rb.velocity = Vector2.zero;

                // Resetta gli stati di sguardo se siamo in IDLE forzato
                if (hero.cState.lookingDown) hero.cState.lookingDown = false;
                if (hero.cState.lookingUp) hero.cState.lookingUp = false;

                if (runningProp != null) try { runningProp.SetValue(hero.cState, false, null); } catch {}

                if (hero.cState.onGround && !hero.cState.attacking && !hero.cState.dashing)
                {
                   PlayAnimSafe("Idle");
                }
            }
        }

        private void PlayAnimSafe(string animName)
        {
            try {
                if (animator == null) animator = hero.GetComponent<tk2dSpriteAnimator>();
                if (animator != null) {
                    // Controlla se sta già suonando per evitare il restart (flickering)
                    if (!animator.IsPlaying(animName))
                    {
                        animator.Play(animName);
                    }
                }
            } catch {}
        }
    }

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
            LeftControl = AddControl(InputControlType.DPadLeft, "Left");
            RightControl = AddControl(InputControlType.DPadRight, "Right");
            UpControl = AddControl(InputControlType.DPadUp, "Up");
            DownControl = AddControl(InputControlType.DPadDown, "Down");
            JumpControl = AddControl(InputControlType.Action1, "Jump (A)");
            CastControl = AddControl(InputControlType.Action2, "Cast (B)");
            AttackControl = AddControl(InputControlType.Action3, "Attack (X)");
            DashControl = AddControl(InputControlType.RightTrigger, "Dash (RT)");
        }
    }

    public class ActionExecutor
    {
        private HeroController hero;
        private SyntheticSoulInputDevice virtualDevice;
        private HeroPhysicsOverride physicsOverride;
        private bool deviceAttached = false;
        private bool actionsBindingComplete = false;

        private DateTime leftExpiration, rightExpiration, upExpiration, downExpiration;
        private DateTime jumpExpiration, attackExpiration, dashExpiration, castExpiration;

        private const double MOVEMENT_DURATION = 0.2;
        private const double TAP_DURATION = 0.05;

        private float lastJumpTime = 0f;
        private float lastAttackTime = 0f;
        private float lastDashTime = 0f;
        private float lastCastTime = 0f;

        // TIMER PER DEBOUNCE
        private float idleWaitTimer = 0f;
        private const float IDLE_DELAY_THRESHOLD = 0.05f; // Ridotto leggermente per reattività

        private object inputHandler;
        private object heroActions;
        private FieldInfo moveInputField;
        private const float RUN_SPEED = 8.3f;

        public ActionExecutor()
        {
            DesktopLogger.Log("═══ ACTIONEXECUTOR V43 (ANIM FIX) ═══");
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
                physicsOverride = hero.gameObject.GetComponent<HeroPhysicsOverride>();
                if (physicsOverride == null) physicsOverride = hero.gameObject.AddComponent<HeroPhysicsOverride>();

                var inputHandlerField = typeof(HeroController).GetField("inputHandler", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                if (inputHandlerField != null)
                {
                    inputHandler = inputHandlerField.GetValue(hero);
                    var heroActionsField = inputHandler.GetType().GetField("inputActions", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                    if (heroActionsField != null) {
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
                    }
                }

                moveInputField = typeof(HeroController).GetField("move_input", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                if (moveInputField == null) moveInputField = typeof(HeroController).GetField("moveInput", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);

                actionsBindingComplete = true;
            }
            catch (Exception ex)
            {
                DesktopLogger.LogError($"Bind Error: {ex.Message}");
            }
        }

        private void BindAction(Type type, string name, InputControl control)
        {
            try
            {
                FieldInfo field = type.GetField(name, BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
                if (field != null)
                {
                    var action = field.GetValue(heroActions) as PlayerAction;
                    if (action != null) action.AddBinding(new DeviceBindingSource(control.Target));
                }
            }
            catch { }
        }

        public void ExecuteAction(string action, bool force = false)
        {
            if (hero == null) hero = HeroController.instance;
            if (hero == null || hero.cState.dead) return;

            if (action == "IDLE")
            {
                ReleaseAllKeys();
                return;
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
                        downExpiration = DateTime.MinValue;
                        break;
                    case "DOWN":
                        downExpiration = now.AddSeconds(MOVEMENT_DURATION);
                        upExpiration = DateTime.MinValue;
                        break;
                    case "JUMP": if (timeNow - lastJumpTime > 0.1f) { jumpExpiration = now.AddSeconds(0.15); lastJumpTime = timeNow; } break;
                    case "ATTACK": if (timeNow - lastAttackTime > 0.25f) { attackExpiration = now.AddSeconds(TAP_DURATION); lastAttackTime = timeNow; } break;
                    case "DASH": if (timeNow - lastDashTime > 0.4f) { dashExpiration = now.AddSeconds(TAP_DURATION); lastDashTime = timeNow; } break;
                    case "SPELL": case "CAST": if (PlayerData.instance.MPCharge >= 33 && timeNow - lastCastTime > 0.3f) { castExpiration = now.AddSeconds(TAP_DURATION); lastCastTime = timeNow; } break;
                }
            }
            catch (Exception e) { DesktopLogger.LogError($"Exec Error: {e.Message}"); }
        }

        public void Update()
        {
            if (hero == null) hero = HeroController.instance;
            if (hero == null) return;

            if (!hero.acceptingInput || hero.cState.dead || hero.cState.transitioning)
            {
                ReleaseAllKeys();
                if (physicsOverride != null) physicsOverride.IsActive = false;
                return;
            }

            if (!actionsBindingComplete && deviceAttached) BindDeviceToHeroActions();

            DateTime now = DateTime.Now;
            float dt = Time.deltaTime;
            ulong tick = InputManager.CurrentTick;

            bool wantLeft = now < leftExpiration;
            bool wantRight = now < rightExpiration;
            bool wantUp = now < upExpiration;
            bool wantDown = now < downExpiration;
            bool wantJump = now < jumpExpiration;
            bool wantAttack = now < attackExpiration;
            bool wantDash = now < dashExpiration;
            bool wantCast = now < castExpiration;

            if (wantLeft) wantRight = false;

            if (deviceAttached && virtualDevice != null)
            {
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

            if (physicsOverride != null && !hero.cState.dashing && !hero.cState.recoiling)
            {
                physicsOverride.IsActive = true;

                bool isBusy = wantJump || wantDash || wantAttack || wantCast ||
                              hero.cState.jumping || hero.cState.dashing || hero.cState.attacking;

                if (wantLeft || wantRight || wantDown || wantUp || isBusy)
                {
                    idleWaitTimer = 0f;
                }

                // V43: Impostiamo i flag di sguardo nel PhysicsOverride
                // Questo permette al LateUpdate di forzare l'animazione corretta
                physicsOverride.IsLookingUp = wantUp && !isBusy && !wantLeft && !wantRight;
                physicsOverride.IsLookingDown = wantDown && !isBusy && !wantLeft && !wantRight;

                if (wantLeft)
                {
                    physicsOverride.TargetVelocityX = -RUN_SPEED;
                    if (moveInputField != null) moveInputField.SetValue(hero, -1.0f);
                    physicsOverride.LockVelocity = false;
                    physicsOverride.ForceIdleAnim = false;
                    hero.FaceLeft();
                }
                else if (wantRight)
                {
                    physicsOverride.TargetVelocityX = RUN_SPEED;
                    if (moveInputField != null) moveInputField.SetValue(hero, 1.0f);
                    physicsOverride.LockVelocity = false;
                    physicsOverride.ForceIdleAnim = false;
                    hero.FaceRight();
                }
                else
                {
                    physicsOverride.TargetVelocityX = 0f;
                    if (moveInputField != null) moveInputField.SetValue(hero, 0.0f);

                    if (hero.cState.onGround && !isBusy)
                    {
                        if (wantDown || wantUp)
                        {
                            physicsOverride.LockVelocity = true;
                            // ForceIdleAnim deve essere falso se stiamo guardando su/giù!
                            physicsOverride.ForceIdleAnim = false;
                        }
                        else
                        {
                            idleWaitTimer += dt;
                            if (idleWaitTimer > IDLE_DELAY_THRESHOLD)
                            {
                                physicsOverride.LockVelocity = true;
                                physicsOverride.ForceIdleAnim = true;
                            }
                            else
                            {
                                physicsOverride.LockVelocity = true;
                                physicsOverride.ForceIdleAnim = false;
                            }
                        }
                    }
                    else
                    {
                        physicsOverride.LockVelocity = false;
                        physicsOverride.ForceIdleAnim = false;
                    }
                }
            }
            else if (physicsOverride != null)
            {
                physicsOverride.IsActive = false;
                physicsOverride.ForceIdleAnim = false;
                physicsOverride.LockVelocity = false;
                physicsOverride.IsLookingUp = false;
                physicsOverride.IsLookingDown = false;
            }
        }

        private void ReleaseAllKeys()
        {
            leftExpiration = DateTime.MinValue;
            rightExpiration = DateTime.MinValue;
            upExpiration = DateTime.MinValue;
            downExpiration = DateTime.MinValue;
        }

        public void DestroyDevice()
        {
            if (virtualDevice != null && deviceAttached)
            {
                InputManager.DetachDevice(virtualDevice);
                virtualDevice = null;
                deviceAttached = false;
                actionsBindingComplete = false;
            }
            if (physicsOverride != null)
            {
                GameObject.Destroy(physicsOverride);
                physicsOverride = null;
            }
        }
    }
}