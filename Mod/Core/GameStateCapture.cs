using UnityEngine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SyntheticSoulMod
{
    public class GameStateCapture
    {
        public class Enemy 
        { 
            public string Type;
            public float X; 
            public float Y; 
            public float Distance; 
            public int HP;
        }

        // NUOVA CLASSE: Rappresenta proiettili, onde d'urto o hitbox nemiche
        public class Danger
        {
            public float X;
            public float Y;
            public float VX; // Velocità X (se ha un rigidbody)
            public float VY; // Velocità Y
            public float Distance;
        }

        public class PlayerState
        {
            public float x;
            public float y; 
            public int health; 
            public int maxHealth;
            public float vx; 
            public float vy; 
            public int soul;
            public bool onGround; 
            public bool touchingWall; 
            public bool damageTaken;
            public List<Enemy> enemies;
            public List<Danger> dangers; // LISTA DEI PERICOLI
        }

        private PlayerState currentState = new PlayerState();
        private readonly object stateLock = new object();

        // Metodo aggiornato per accettare anche la lista 'dangers'
        public void UpdatePlayerState(Vector3 pos, int hp, int maxHp, Vector2 vel, int soul, List<Enemy> enemies, List<Danger> dangers, bool dmg, bool ground, bool wall)
        {
            lock (stateLock)
            {
                currentState.x = pos.x;
                currentState.y = pos.y;
                currentState.health = hp; 
                currentState.maxHealth = maxHp;
                currentState.vx = vel.x; 
                currentState.vy = vel.y;
                currentState.soul = soul;
                currentState.enemies = enemies;
                currentState.dangers = dangers; // Aggiornamento lista pericoli
                currentState.damageTaken = dmg;
                currentState.onGround = ground;
                currentState.touchingWall = wall;
            }
        }

        public string GetStateJson()
        {
            lock (stateLock) 
            { 
                StringBuilder sb = new StringBuilder();
                sb.Append("{");
                sb.Append($"\"x\":{currentState.x.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                sb.Append($"\"y\":{currentState.y.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                sb.Append($"\"health\":{currentState.health},");
                sb.Append($"\"maxHealth\":{currentState.maxHealth},");
                sb.Append($"\"vx\":{currentState.vx.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                sb.Append($"\"vy\":{currentState.vy.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                sb.Append($"\"soul\":{currentState.soul},");
                sb.Append($"\"onGround\":{(currentState.onGround ? "true" : "false")},");
                sb.Append($"\"touchingWall\":{(currentState.touchingWall ? "true" : "false")},");
                sb.Append($"\"damageTaken\":{(currentState.damageTaken ? "true" : "false")},");
                
                // ENEMIES LIST
                sb.Append("\"enemies\":[");
                if (currentState.enemies != null && currentState.enemies.Count > 0)
                {
                    for (int i = 0; i < currentState.enemies.Count; i++)
                    {
                        var e = currentState.enemies[i];
                        sb.Append("{");
                        sb.Append($"\"Type\":\"{e.Type}\",");
                        sb.Append($"\"X\":{e.X.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                        sb.Append($"\"Y\":{e.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                        sb.Append($"\"Distance\":{e.Distance.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                        sb.Append($"\"HP\":{e.HP}");
                        sb.Append("}");
                        if (i < currentState.enemies.Count - 1) sb.Append(",");
                    }
                }
                sb.Append("],"); // Virgola importante

                // DANGERS LIST (Nuova sezione)
                sb.Append("\"dangers\":[");
                if (currentState.dangers != null && currentState.dangers.Count > 0)
                {
                    for (int i = 0; i < currentState.dangers.Count; i++)
                    {
                        var d = currentState.dangers[i];
                        sb.Append("{");
                        sb.Append($"\"X\":{d.X.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                        sb.Append($"\"Y\":{d.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                        sb.Append($"\"VX\":{d.VX.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                        sb.Append($"\"VY\":{d.VY.ToString(System.Globalization.CultureInfo.InvariantCulture)},");
                        sb.Append($"\"Distance\":{d.Distance.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                        sb.Append("}");
                        if (i < currentState.dangers.Count - 1) sb.Append(",");
                    }
                }
                sb.Append("]");

                sb.Append("}");
                return sb.ToString();
            }
        }
    }
}