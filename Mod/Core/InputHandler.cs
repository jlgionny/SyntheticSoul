using UnityEngine;
using System;

namespace SyntheticSoulMod
{
    public class InputHandler
    {
        public bool isMovingLeft = false;
        public bool isMovingRight = false;
        public bool jumpRequested = false;
        public bool attackRequested = false;
        public bool dashRequested = false;
        public bool focusRequested = false;

        public enum AIAction
        {
            IDLE = 0, JUMP = 1, ATTACK = 2, MOVE_LEFT = 3, MOVE_RIGHT = 4, DASH = 5, FOCUS = 6
        }

        public static AIAction ParseAction(string actionString)
        {
            try { return (AIAction)Enum.Parse(typeof(AIAction), actionString, true); }
            catch { return AIAction.IDLE; }
        }

        public void ExecuteAction(AIAction action)
        {
            // Reset preventivo
            ResetState();

            switch (action)
            {
                case AIAction.MOVE_LEFT: isMovingLeft = true; break;
                case AIAction.MOVE_RIGHT: isMovingRight = true; break;
                case AIAction.JUMP: jumpRequested = true; break;
                case AIAction.ATTACK: attackRequested = true; break;
                case AIAction.DASH: dashRequested = true; break;
                case AIAction.FOCUS: focusRequested = true; break;
            }
        }

        //Azzera tutto immediatamente
        public void ResetState()
        {
            isMovingLeft = false;
            isMovingRight = false;
            jumpRequested = false;
            attackRequested = false;
            dashRequested = false;
            focusRequested = false;
        }

        public float GetHorizontalInput()
        {
            if (isMovingLeft) return -1f;
            if (isMovingRight) return 1f;
            return 0f;
        }
    }
}