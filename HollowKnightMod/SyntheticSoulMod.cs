using Modding;
using System;
using System.Collections.Generic;
using UnityEngine;

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

        public override string GetVersion() => "1.0.0.0";

        public override void Initialize(Dictionary<string, Dictionary<string, GameObject>> preloadedObjects)
        {
            Log("Initializing SyntheticSoul Mod...");
            DesktopLogger.Log("=== SYNTHETIC SOUL MOD INITIALIZING ===");

            stateExtractor = new GameStateExtractor();
            actionExecutor = new ActionExecutor();
            communicator = new SocketCommunicator(PORT);

            ModHooks.HeroUpdateHook += OnHeroUpdate;
            ModHooks.AfterTakeDamageHook += OnTakeDamage;

            Log("SyntheticSoul Mod initialized! Starting persistent server...");
            DesktopLogger.Log("Starting persistent socket server on port 5555...");

            try
            {
                communicator.StartServer();
                isTraining = true;
                Log("Persistent server started! Waiting for Python connections...");
                DesktopLogger.Log("✓ Persistent server thread spawned successfully");
            }
            catch (Exception e)
            {
                LogError($"Failed to start server: {e.Message}");
                DesktopLogger.LogError($"Server start failed: {e.Message}\n{e.StackTrace}");
            }

            Log("SyntheticSoul Mod ready!");
            DesktopLogger.Log("SyntheticSoul Mod initialization complete");
        }

        private void OnHeroUpdate()
        {
            if (hero == null)
                hero = HeroController.instance;

            if (!isTraining)
                return;

            // FIX: Detect connection status changes
            bool currentlyConnected = communicator.IsConnected;

            if (currentlyConnected != wasConnected)
            {
                if (currentlyConnected)
                {
                    Log("[SyntheticSoul] ✓ Python agent connected!");
                    DesktopLogger.Log($"=== PYTHON AGENT CONNECTED at {DateTime.Now:HH:mm:ss} ===");

                    // FIX: Destroy old device before creating new ActionExecutor
                    if (actionExecutor != null)
                    {
                        actionExecutor.DestroyDevice();
                    }

                    actionExecutor = new ActionExecutor();
                }
                else
                {
                    Log("[SyntheticSoul] ✗ Python agent disconnected. Waiting for reconnection...");
                    DesktopLogger.LogWarning($"Python agent disconnected at {DateTime.Now:HH:mm:ss}");

                    // FIX: CRITICAL - Clean up device on disconnect to restore keyboard
                    if (actionExecutor != null)
                    {
                        actionExecutor.DestroyDevice();
                        DesktopLogger.Log("✓ Virtual device cleaned up - keyboard control restored");
                    }
                }

                wasConnected = currentlyConnected;
            }

            // Only process AI if connected
            if (!currentlyConnected)
                return;

            // Update action executor every frame
            if (actionExecutor != null)
            {
                actionExecutor.Update();
            }

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
                if (hero != null && hero.cState.transitioning)
                    return;

                var gameState = stateExtractor.ExtractState();

                if (gameState.isDead)
                {
                    DesktopLogger.LogWarning("Knight is DEAD - sending dead state to Python");
                }

                communicator.SendState(gameState);
                string action = communicator.ReceiveAction();

                bool canExecute = hero != null && !hero.cState.dead && !hero.cState.recoiling;

                if (!string.IsNullOrEmpty(action) && action != "IDLE" && canExecute)
                {
                    actionExecutor.ExecuteAction(action);
                }
            }
            catch (Exception e)
            {
                LogError($"[SyntheticSoul] Error in AI step: {e.Message}\n{e.StackTrace}");
                DesktopLogger.LogError($"AI STEP CRASH: {e.Message}\n{e.StackTrace}");
            }
        }

        private int OnTakeDamage(int hazardType, int damage)
        {
            int currentHealth = PlayerData.instance != null ? PlayerData.instance.health : -1;
            Log($"[SyntheticSoul] Knight took {damage} damage (type:{hazardType})! Health: {currentHealth}");
            DesktopLogger.Log($"DAMAGE TAKEN: {damage} dmg, HP: {currentHealth}, HazardType: {hazardType}");
            return damage;
        }

        public void Unload()
        {
            Log("Unloading SyntheticSoul Mod...");
            DesktopLogger.Log("=== SYNTHETIC SOUL MOD UNLOADING ===");

            // FIX: CRITICAL - Clean up virtual device FIRST
            if (actionExecutor != null)
            {
                DesktopLogger.Log("Destroying ActionExecutor and virtual device...");
                actionExecutor.DestroyDevice();
                actionExecutor = null;
                DesktopLogger.Log("✓ ActionExecutor cleanup complete");
            }

            ModHooks.HeroUpdateHook -= OnHeroUpdate;
            ModHooks.AfterTakeDamageHook -= OnTakeDamage;

            if (communicator != null)
            {
                communicator.Close();
            }

            isTraining = false;

            Log("SyntheticSoul Mod unloaded!");
            DesktopLogger.Log("=== SYNTHETIC SOUL MOD UNLOADED - KEYBOARD CONTROL RESTORED ===");
        }
    }
}