using UnityEngine;
using System;

namespace SyntheticSoulMod
{
    public class InputHandler
    {
        // Variabili di stato semplici
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

        public void ExecuteAction(AIAction action)
        {
            // Reset direzioni per evitare conflitti (sinistra+destra insieme)
            if (action == AIAction.IDLE || action == AIAction.JUMP || action == AIAction.ATTACK)
            {
                // Non resettiamo il movimento se saltiamo/attacchiamo, così possiamo farlo mentre corriamo
                if (action == AIAction.IDLE) 
                {
                    isMovingLeft = false;
                    isMovingRight = false;
                }
            }
            else if (action == AIAction.MOVE_LEFT)
            {
                isMovingLeft = true;
                isMovingRight = false;
            }
            else if (action == AIAction.MOVE_RIGHT)
            {
                isMovingRight = true;
                isMovingLeft = false;
            }

            switch (action)
            {
                case AIAction.JUMP: jumpRequested = true; break;
                case AIAction.ATTACK: attackRequested = true; break;
                case AIAction.DASH: dashRequested = true; break;
                case AIAction.FOCUS: focusRequested = true; break;
            }
        }

        public static AIAction ParseAction(string actionString)
        {
            if (Enum.TryParse(actionString.ToUpper(), out AIAction action)) return action;
            return AIAction.IDLE;
        }

        public float GetHorizontalInput()
        {
            if (isMovingLeft) return -1f;
            if (isMovingRight) return 1f;
            return 0f;
        }
    }
}