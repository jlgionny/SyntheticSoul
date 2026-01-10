using UnityEngine;
using System;
using System.Collections.Generic;
using System.Text; // Serve per StringBuilder

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

        public class PlayerState
        {
            public float x; public float y; public int health; public int maxHealth;
            public float vx; public float vy; public int soul;
            public bool onGround; public bool touchingWall; public bool damageTaken;
            public List<Enemy> enemies;
        }

        private PlayerState currentState = new PlayerState();
        private readonly object stateLock = new object();

        public void UpdatePlayerState(Vector3 pos, int hp, int maxHp, Vector2 vel, int soul, List<Enemy> enemies, bool dmg, bool ground, bool wall)
        {
            lock (stateLock)
            {
                currentState.x = pos.x; currentState.y = pos.y;
                currentState.health = hp; currentState.maxHealth = maxHp;
                currentState.vx = vel.x; currentState.vy = vel.y;
                currentState.soul = soul;
                currentState.enemies = enemies;
                currentState.damageTaken = dmg;
                currentState.onGround = ground;
                currentState.touchingWall = wall;
            }
        }

        public string GetStateJson()
        {
            lock (stateLock) 
            { 
                // Costruzione manuale del JSON per evitare errori di compilazione e dipendenze mancanti
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
                sb.Append("]");
                sb.Append("}");
                return sb.ToString();
            }
        }
    }
}