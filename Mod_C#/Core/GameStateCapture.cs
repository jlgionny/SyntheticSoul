using UnityEngine;
using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Gestisce lo stato del gioco (Player, Nemici, Ambiente).
    /// Thread-Safe: Usa un lock per gestire la lettura/scrittura concorrente.
    /// </summary>
    public class GameStateCapture
    {
        // Struttura dati per un singolo nemico
        public class Enemy
        {
            [JsonProperty("type")] public string Type { get; set; }
            [JsonProperty("x")] public float X { get; set; }
            [JsonProperty("y")] public float Y { get; set; }
            [JsonProperty("distance")] public float Distance { get; set; }
            [JsonProperty("hp")] public int HP { get; set; }
        }

        // Struttura completa dello stato inviato via JSON
        public class GameState
        {
            [JsonProperty("player_x")] public float PlayerX { get; set; }
            [JsonProperty("player_y")] public float PlayerY { get; set; }
            [JsonProperty("player_hp")] public int PlayerHP { get; set; }
            [JsonProperty("player_max_hp")] public int PlayerMaxHP { get; set; }
            [JsonProperty("player_soul")] public int PlayerSoul { get; set; }
            [JsonProperty("player_max_soul")] public int PlayerMaxSoul { get; set; }
            [JsonProperty("player_vel_x")] public float PlayerVelX { get; set; }
            [JsonProperty("player_vel_y")] public float PlayerVelY { get; set; }
            [JsonProperty("enemies")] public List<Enemy> Enemies { get; set; }
            [JsonProperty("has_ground_below")] public bool HasGroundBelow { get; set; }
            [JsonProperty("touching_wall_left")] public bool TouchingWallLeft { get; set; }
            [JsonProperty("touching_wall_right")] public bool TouchingWallRight { get; set; }
            [JsonProperty("took_damage")] public bool TookDamage { get; set; }
            [JsonProperty("damage_malus_active")] public bool DamageMalusActive { get; set; }
            [JsonProperty("timestamp")] public long Timestamp { get; set; }
        }

        private GameState currentState = new GameState();
        
        // Oggetto usato per sincronizzare i thread (Lock)
        private readonly object _stateLock = new object();

        /// <summary>
        /// Aggiorna i dati. Chiamato dal Main Thread di Unity.
        /// </summary>
        public void UpdatePlayerState(Vector3 position, int hp, int maxHP, Vector2 velocity, int soul = 0, int maxSoul = 99,
            List<Enemy> enemies = null, bool groundBelow = false, bool touchingWallLeft = false, bool touchingWallRight = false,
            bool tookDamage = false, bool damageMalusActive = false)
        {
            // BLOCCA l'accesso agli altri thread mentre scriviamo
            lock (_stateLock)
            {
                currentState.PlayerX = position.x;
                currentState.PlayerY = position.y;
                currentState.PlayerHP = hp;
                currentState.PlayerMaxHP = maxHP;
                currentState.PlayerSoul = soul;
                currentState.PlayerMaxSoul = maxSoul;
                currentState.PlayerVelX = velocity.x;
                currentState.PlayerVelY = velocity.y;
                currentState.Enemies = enemies ?? new List<Enemy>();
                currentState.HasGroundBelow = groundBelow;
                currentState.TouchingWallLeft = touchingWallLeft;
                currentState.TouchingWallRight = touchingWallRight;
                currentState.TookDamage = tookDamage;
                currentState.DamageMalusActive = damageMalusActive;
                currentState.Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            }
        }

        /// <summary>
        /// Restituisce lo stato serializzato in JSON. Chiamato dal Thread del Server.
        /// </summary>
        public string GetStateAsJSON()
        {
            // BLOCCA l'accesso mentre leggiamo e convertiamo in JSON
            lock (_stateLock)
            {
                return JsonConvert.SerializeObject(currentState);
            }
        }
    }
}