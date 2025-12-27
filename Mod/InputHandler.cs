using UnityEngine;
using System;
using System.Collections.Generic;
using System.Reflection;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Gestisce gli input ricevuti dall'IA via socket.
    /// Converte stringhe di azioni (JUMP, ATTACK, MOVE_LEFT, etc.) in input reali nel gioco.
    /// </summary>
    public class InputHandler
    {
        private HeroController heroController;
        
        // Dizionario di azioni disponibili
        public enum AIAction
        {
            IDLE = 0,           // Nessun input
            JUMP = 1,           // Salto
            ATTACK = 2,         // Attacco (Nail)
            MOVE_LEFT = 3,      // Movimento sinistra
            MOVE_RIGHT = 4,     // Movimento destra
            DASH = 5,           // Dash (se disponibile)
            FOCUS = 6           // Focus Soul per curare
        }

        public InputHandler(HeroController hero)
        {
            this.heroController = hero;
        }

        /// <summary>
        /// Esegue un'azione ricevuta dall'IA.
        /// </summary>
        public void ExecuteAction(AIAction action)
        {
            if (heroController == null) return;

            switch (action)
            {
                case AIAction.JUMP:
                    PerformJump();
                    break;
                case AIAction.ATTACK:
                    PerformAttack();
                    break;
                case AIAction.MOVE_LEFT:
                    Move(-1);
                    break;
                case AIAction.MOVE_RIGHT:
                    Move(1);
                    break;
                case AIAction.DASH:
                    PerformDash();
                    break;
                case AIAction.FOCUS:
                    PerformFocus();
                    break;
                case AIAction.IDLE:
                default:
                    // Nessun input
                    break;
            }
        }

        /// <summary>
        /// Converte una stringa in un'azione.
        /// Utile per ricevere azioni dal socket.
        /// </summary>
        public static AIAction ParseAction(string actionString)
        {
            if (Enum.TryParse(actionString.ToUpper(), out AIAction action))
            {
                return action;
            }
            return AIAction.IDLE;
        }

        private void PerformJump()
        {
            try
            {
                // Accedi al metodo Jump di HeroController tramite Reflection
                MethodInfo jumpMethod = typeof(HeroController).GetMethod(
                    "Jump",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null,
                    Type.EmptyTypes,
                    null
                );

                if (jumpMethod != null)
                {
                    jumpMethod.Invoke(heroController, null);
                    Modding.Logger.Log("[AI Action] JUMP eseguito");
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[AI Action] Errore Jump: {e.Message}");
            }
        }

        private void PerformAttack()
        {
            try
            {
                // Accedi al metodo Attack di HeroController
                MethodInfo attackMethod = typeof(HeroController).GetMethod(
                    "Attack",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null,
                    Type.EmptyTypes,
                    null
                );

                if (attackMethod != null)
                {
                    attackMethod.Invoke(heroController, null);
                    Modding.Logger.Log("[AI Action] ATTACK eseguito");
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[AI Action] Errore Attack: {e.Message}");
            }
        }

        private void Move(int direction)
        {
            try
            {
                // Accedi al campo moveInput di HeroController
                FieldInfo moveInputField = typeof(HeroController).GetField(
                    "moveInput",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                );

                if (moveInputField != null)
                {
                    moveInputField.SetValue(heroController, (float)direction);
                    Modding.Logger.Log($"[AI Action] MOVE {(direction > 0 ? "RIGHT" : "LEFT")} impostato");
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[AI Action] Errore Move: {e.Message}");
            }
        }

        private void PerformDash()
        {
            try
            {
                // Accedi al metodo Dash di HeroController (se esiste)
                MethodInfo dashMethod = typeof(HeroController).GetMethod(
                    "Dash",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null,
                    Type.EmptyTypes,
                    null
                );

                if (dashMethod != null)
                {
                    dashMethod.Invoke(heroController, null);
                    Modding.Logger.Log("[AI Action] DASH eseguito");
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[AI Action] Errore Dash: {e.Message}");
            }
        }

        private void PerformFocus()
        {
            try
            {
                // Accedi al metodo Focus di HeroController (per curare con Soul)
                MethodInfo focusMethod = typeof(HeroController).GetMethod(
                    "CastSpell",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                );

                if (focusMethod != null)
                {
                    focusMethod.Invoke(heroController, null);
                    Modding.Logger.Log("[AI Action] FOCUS (Heal) eseguito");
                }
            }
            catch (Exception e)
            {
                Modding.Logger.Log($"[AI Action] Errore Focus: {e.Message}");
            }
        }

        /// <summary>
        /// Resetta tutti gli input (chiamato ogni frame per evitare input "bloccati")
        /// </summary>
        public void ResetInputs()
        {
            try
            {
                FieldInfo moveInputField = typeof(HeroController).GetField(
                    "moveInput",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                );

                if (moveInputField != null)
                {
                    moveInputField.SetValue(heroController, 0f);
                }
            }
            catch { }
        }
    }
}
