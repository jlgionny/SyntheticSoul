using UnityEngine;
using System;
using Modding;

namespace SyntheticSoulMod
{
    public class InputHandler
    {
        private DateTime jumpExpiration = DateTime.MinValue;
        private DateTime attackExpiration = DateTime.MinValue;
        private DateTime dashExpiration = DateTime.MinValue;
        private DateTime castExpiration = DateTime.MinValue;
        
        private DateTime moveLeftExpiration = DateTime.MinValue;
        private DateTime moveRightExpiration = DateTime.MinValue;
        private DateTime lookUpExpiration = DateTime.MinValue;
        private DateTime lookDownExpiration = DateTime.MinValue;
        
        public readonly object inputLock = new object();

        // Timer
        private const double JUMP_MIN_DURATION = 0.2; // Ridotto leggermente per reattività
        private const double SPELL_MIN_DURATION = 0.25;
        private const double MOVEMENT_TIMEOUT = 0.2;   

        public enum AIAction
        {
            IDLE = 0, 
            MOVE_LEFT = 1, 
            MOVE_RIGHT = 2, 
            JUMP = 3, 
            DASH = 4, 
            ATTACK = 5, 
            ATTACK_UP = 6, 
            ATTACK_DOWN = 7,
            CAST_NEUTRAL = 8 
            // RIMOSSI: CAST_UP, CAST_DOWN, FOCUS
        }

        public static AIAction ParseAction(string actionString)
        {
            try { return (AIAction)Enum.Parse(typeof(AIAction), actionString, true); }
            catch { return AIAction.IDLE; }
        }

        public bool jumpHeld { get { lock(inputLock) { return DateTime.Now < jumpExpiration; } } }
        public bool attackHeld { get { lock(inputLock) { return DateTime.Now < attackExpiration; } } }
        public bool dashHeld { get { lock(inputLock) { return DateTime.Now < dashExpiration; } } }
        public bool castHeld { get { lock(inputLock) { return DateTime.Now < castExpiration; } } }
        
        // RIMOSSO: focusHeld

        public bool isMovingLeft { get { lock(inputLock) { return DateTime.Now < moveLeftExpiration; } } }
        public bool isMovingRight { get { lock(inputLock) { return DateTime.Now < moveRightExpiration; } } }
        public bool isLookingUp { get { lock(inputLock) { return DateTime.Now < lookUpExpiration; } } }
        public bool isLookingDown { get { lock(inputLock) { return DateTime.Now < lookDownExpiration; } } }

        public void ExecuteAction(AIAction action)
        {
            lock (inputLock)
            {
                DateTime now = DateTime.Now;
                
                if (action == AIAction.JUMP) jumpExpiration = now.AddSeconds(JUMP_MIN_DURATION);
                if (action == AIAction.DASH) dashExpiration = now.AddSeconds(0.15);
                
                if (action == AIAction.ATTACK || action == AIAction.ATTACK_UP || action == AIAction.ATTACK_DOWN) 
                    attackExpiration = now.AddSeconds(0.15);
                
                // Ora gestisce solo CAST_NEUTRAL
                if (action == AIAction.CAST_NEUTRAL) 
                    castExpiration = now.AddSeconds(SPELL_MIN_DURATION);

                if (action == AIAction.MOVE_LEFT) 
                {
                    moveLeftExpiration = now.AddSeconds(MOVEMENT_TIMEOUT);
                    moveRightExpiration = DateTime.MinValue;
                }
                else if (action == AIAction.MOVE_RIGHT) 
                {
                    moveRightExpiration = now.AddSeconds(MOVEMENT_TIMEOUT);
                    moveLeftExpiration = DateTime.MinValue;
                }

                // Qui manteniamo SOLO ATTACK_UP/DOWN per guardare su/giù
                if (action == AIAction.ATTACK_UP) 
                {
                    lookUpExpiration = now.AddSeconds(MOVEMENT_TIMEOUT);
                    lookDownExpiration = DateTime.MinValue;
                }
                else if (action == AIAction.ATTACK_DOWN) 
                {
                    lookDownExpiration = now.AddSeconds(MOVEMENT_TIMEOUT);
                    lookUpExpiration = DateTime.MinValue;
                }
            }
        }

        public void ResetState()
        {
            lock (inputLock)
            {
                jumpExpiration = DateTime.MinValue;
                attackExpiration = DateTime.MinValue;
                dashExpiration = DateTime.MinValue;
                castExpiration = DateTime.MinValue;
                moveLeftExpiration = DateTime.MinValue;
                moveRightExpiration = DateTime.MinValue;
                lookUpExpiration = DateTime.MinValue;
                lookDownExpiration = DateTime.MinValue;
            }
        }

        public float GetHorizontalInput() { lock (inputLock) { if (DateTime.Now < moveLeftExpiration) return -1f; if (DateTime.Now < moveRightExpiration) return 1f; return 0f; } }
        public float GetVerticalInput() { lock (inputLock) { if (DateTime.Now < lookDownExpiration) return -1f; if (DateTime.Now < lookUpExpiration) return 1f; return 0f; } }
    }
}