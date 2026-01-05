using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace SyntheticSoulMod
{
    public class GameStateCapture
    {
        public struct Enemy
        {
            [JsonProperty("type")] public string Type;
            [JsonProperty("x")] public float X;
            [JsonProperty("y")] public float Y;
            [JsonProperty("dist")] public float Distance;
            [JsonProperty("hp")] public int HP;
        }

        public class GameState
        {
            [JsonProperty("p_x")] public float PlayerX;
            [JsonProperty("p_y")] public float PlayerY;
            [JsonProperty("hp")] public int HP;
            [JsonProperty("max_hp")] public int MaxHP;
            [JsonProperty("soul")] public int Soul;
            [JsonProperty("vel_x")] public float VelX;
            [JsonProperty("vel_y")] public float VelY;
            [JsonProperty("ground")] public bool OnGround;
            [JsonProperty("wall")] public bool TouchingWall;
            [JsonProperty("hurt")] public bool TookDamage; 
            [JsonProperty("enemies")] public List<Enemy> Enemies;
        }

        private GameState currentState = new GameState();
        private readonly object _lock = new object();

        public void UpdatePlayerState(Vector3 pos, int hp, int maxHp, Vector2 vel, int soul, 
                                      List<Enemy> enemies, bool tookDamage, bool onGround, bool touchingWall)
        {
            lock (_lock)
            {
                currentState.PlayerX = pos.x;
                currentState.PlayerY = pos.y;
                currentState.HP = hp;
                currentState.MaxHP = maxHp;
                currentState.VelX = vel.x;
                currentState.VelY = vel.y;
                currentState.Soul = soul;
                currentState.Enemies = enemies ?? new List<Enemy>();
                currentState.TookDamage = tookDamage;
                currentState.OnGround = onGround;
                currentState.TouchingWall = touchingWall;
            }
        }

        public string GetJson()
        {
            lock (_lock)
            {
                return JsonConvert.SerializeObject(currentState);
            }
        }
        
        // Metodo legacy per compatibilitÃ col Server
        public string GetStateAsJSON()
        {
            return GetJson();
        }
    }
}