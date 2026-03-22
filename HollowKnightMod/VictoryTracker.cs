using System;
using System.Collections;
using UnityEngine;
using HutongGames.PlayMaker;

namespace SyntheticSoulMod
{
    /// <summary>
    /// Sistema robusto di rilevamento e tracking vittoria per Hall of Gods.
    ///
    /// PROBLEMA ORIGINALE:
    /// Quando l'agente sconfigge le mantidi nella HoG, il contatore vittorie non viene
    /// aggiornato perché la scena viene resettata prima che il gioco registri la vittoria.
    /// La sequenza di vittoria di HoG include:
    ///   1. Ultimo boss muore (HP=0, evento Die)
    ///   2. Animazione di morte (~1-2s)
    ///   3. Schermo di vittoria / statua che cambia stato
    ///   4. Aggiornamento PlayerData (bossRushKills, record, ecc.)
    ///   5. Transizione scena
    ///
    /// SOLUZIONE:
    /// Questo tracker intercetta la vittoria al punto 1, aspetta che i dati vengano
    /// scritti (punto 4), e solo DOPO inizia il reload per il training.
    ///
    /// METRICHE TRACKED:
    /// - Vittorie totali (persistenti in sessione)
    /// - Tempo di vittoria (durata episodio)
    /// - Win rate (vittorie/episodi totali)
    /// - Streak corrente
    /// </summary>
    public class VictoryTracker
    {
        // ========== CONFIGURAZIONE ==========

        /// <summary>Tempo di attesa dopo l'ultima kill per permettere al gioco di registrare la vittoria.</summary>
        public float VictoryWaitTime { get; set; } = 2.5f;

        /// <summary>Tempo massimo di attesa per la conferma di vittoria prima di forzare il reload.</summary>
        public float VictoryTimeout { get; set; } = 8.0f;

        /// <summary>Se true, aspetta che PlayerData confermi la vittoria prima di resettare.</summary>
        public bool WaitForPlayerDataConfirmation { get; set; } = true;

        // ========== STATO TRACKING ==========

        // Contatori sessione
        public int TotalWins { get; private set; } = 0;
        public int TotalEpisodes { get; private set; } = 0;
        public int CurrentStreak { get; private set; } = 0;
        public int BestStreak { get; private set; } = 0;
        public float WinRate => TotalEpisodes > 0 ? (float)TotalWins / TotalEpisodes : 0f;

        // Stato episodio corrente
        public bool IsVictoryInProgress { get; private set; } = false;
        public bool IsVictoryConfirmed { get; private set; } = false;

        private float episodeStartTime = 0f;
        private float lastVictoryTime = 0f;
        private float lastVictoryDuration = 0f;

        // Stato HoG specifico
        private int hogBossLevelBefore = -1;  // Livello statua pre-fight
        private bool hogDataCaptured = false;

        // ========== EVENTS ==========

        /// <summary>Chiamato quando la vittoria è confermata e il gioco ha registrato i dati.</summary>
        public event Action<VictoryResult> OnVictoryConfirmed;

        /// <summary>Chiamato quando l'episodio finisce (sia vittoria che sconfitta).</summary>
        public event Action<EpisodeResult> OnEpisodeEnd;

        public class VictoryResult
        {
            public float duration;           // Durata episodio in secondi
            public int mantisKilled;         // Mantidi uccise
            public int totalWins;            // Vittorie totali sessione
            public int streak;               // Streak corrente
            public bool hogDataUpdated;      // Se HoG ha aggiornato i record
        }

        public class EpisodeResult
        {
            public bool isVictory;
            public float duration;
            public int mantisKilled;
            public float bossHpRemaining;
        }

        // ========== API PUBBLICA ==========

        /// <summary>
        /// Chiamare all'inizio di ogni episodio (dopo il reset della scena).
        /// </summary>
        public void OnEpisodeStart()
        {
            episodeStartTime = Time.time;
            IsVictoryInProgress = false;
            IsVictoryConfirmed = false;
            hogDataCaptured = false;

            // Cattura lo stato HoG pre-fight per confronto post-vittoria
            CaptureHoGStateBefore();

            TotalEpisodes++;
            DesktopLogger.Log($"[Victory] Episode {TotalEpisodes} started | Wins: {TotalWins} | WR: {WinRate:P1} | Streak: {CurrentStreak}");
        }

        /// <summary>
        /// Chiamare quando una mantide muore. Restituisce true se tutte le mantidi
        /// sono state sconfitte e la vittoria è iniziata.
        /// </summary>
        public bool OnMantisKilled(int totalKilled)
        {
            if (totalKilled >= 3 && !IsVictoryInProgress)
            {
                IsVictoryInProgress = true;
                lastVictoryTime = Time.time;
                lastVictoryDuration = Time.time - episodeStartTime;

                DesktopLogger.Log($"[Victory] ★★★ ALL MANTIS LORDS DEFEATED! ★★★");
                DesktopLogger.Log($"[Victory] Duration: {lastVictoryDuration:F1}s | Starting victory sequence...");

                return true;
            }
            return false;
        }


        /// <summary>
        /// Coroutine da avviare quando la vittoria viene rilevata.
        /// Aspetta che il gioco registri la vittoria, restituisce quando è sicuro resettare.
        /// </summary>
        public IEnumerator WaitForVictoryConfirmation()
        {
            if (!IsVictoryInProgress)
            {
                DesktopLogger.LogWarning("[Victory] WaitForVictory called but no victory in progress!");
                yield break;
            }

            DesktopLogger.Log("═══════════════════════════════════════");
            DesktopLogger.Log("═══    VICTORY SEQUENCE STARTED    ═══");
            DesktopLogger.Log("═══════════════════════════════════════");

            float startWait = Time.time;
            bool playerDataConfirmed = false;

            // FASE 1: Attesa minima per l'animazione di morte
            DesktopLogger.Log($"[Victory] Phase 1: Waiting {VictoryWaitTime:F1}s for death animation...");
            yield return new WaitForSeconds(VictoryWaitTime);

            // FASE 2: Verifica PlayerData (SOLO se abilitato)
            if (WaitForPlayerDataConfirmation)
            {
                DesktopLogger.Log("[Victory] Phase 2: Waiting for PlayerData confirmation...");
                float phase2Start = Time.time;
                float phase2Timeout = VictoryTimeout - VictoryWaitTime;

                while (Time.time - phase2Start < phase2Timeout)
                {
                    if (CheckHoGVictoryRegistered())
                    {
                        playerDataConfirmed = true;
                        DesktopLogger.Log("[Victory] ✓ PlayerData confirmed victory!");
                        break;
                    }

                    if (CheckBattleControlVictory())
                    {
                        playerDataConfirmed = true;
                        DesktopLogger.Log("[Victory] ✓ Battle Control FSM confirmed victory!");
                        break;
                    }

                    yield return new WaitForSeconds(0.1f);
                }

                if (!playerDataConfirmed)
                {
                    DesktopLogger.LogWarning("[Victory] ⚠ PlayerData confirmation timeout");
                    playerDataConfirmed = false;
                }
            }

            // FASE 3: Conferma vittoria e aggiorna metriche
            IsVictoryConfirmed = true;
            TotalWins++;
            CurrentStreak++;
            BestStreak = Mathf.Max(BestStreak, CurrentStreak);

            float totalWaitTime = Time.time - startWait;
            var result = new VictoryResult
            {
                duration = lastVictoryDuration,
                mantisKilled = 3,
                totalWins = TotalWins,
                streak = CurrentStreak,
                hogDataUpdated = playerDataConfirmed
            };

            DesktopLogger.Log("═══════════════════════════════════════");
            DesktopLogger.Log($"[Victory] ✓✓✓ VICTORY CONFIRMED (waited {totalWaitTime:F1}s)");
            DesktopLogger.Log($"[Victory] Wins: {TotalWins} | WR: {WinRate:P1} | Streak: {CurrentStreak} | Best: {BestStreak}");
            DesktopLogger.Log($"[Victory] HoG Data Updated: {playerDataConfirmed}");
            DesktopLogger.Log("═══════════════════════════════════════");

            OnVictoryConfirmed?.Invoke(result);
        }



        /// <summary>
        /// Chiamare quando il player muore (sconfitta).
        /// </summary>
        public void OnDefeat()
        {
            CurrentStreak = 0;

            var result = new EpisodeResult
            {
                isVictory = false,
                duration = Time.time - episodeStartTime,
                mantisKilled = 0,  // Sarà aggiornato dal chiamante
                bossHpRemaining = 0
            };

            OnEpisodeEnd?.Invoke(result);
        }

        /// <summary>
        /// Resetta tutti i contatori (per nuova sessione di training).
        /// </summary>
        public void ResetAll()
        {
            TotalWins = 0;
            TotalEpisodes = 0;
            CurrentStreak = 0;
            BestStreak = 0;
            IsVictoryInProgress = false;
            IsVictoryConfirmed = false;
            hogDataCaptured = false;
            DesktopLogger.Log("[Victory] All counters reset");
        }

        // ========== LOGICA INTERNA HoG ==========

        private void CaptureHoGStateBefore()
        {
            try
            {
                var pd = PlayerData.instance;
                if (pd == null) return;

                // Hall of Gods tracka le vittorie nelle variabili:
                // - bossRushMode (bool)
                // - completedBossRushCycles (int, per Pantheons) -- non per singoli boss
                // - I boss hanno statue con livelli (0=non battuto, 1=Attuned, 2=Ascended, 3=Radiant)

                // La variabile specifica per le Mantis Lords nella HoG è nel BossStatue
                // ma possiamo anche controllare se il completamento viene registrato
                // guardando la FSM della statua o il BossSequenceController.

                hogBossLevelBefore = GetCurrentBossStatueLevel();
                hogDataCaptured = true;

                if (hogBossLevelBefore >= 0)
                    DesktopLogger.Log($"[Victory] HoG statue level before fight: {hogBossLevelBefore}");
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"[Victory] CaptureHoGState error: {e.Message}");
            }
        }

        private bool CheckHoGVictoryRegistered()
        {
            try
            {
                if (!hogDataCaptured) return false;

                int currentLevel = GetCurrentBossStatueLevel();
                // Se il livello è aumentato, la vittoria è stata registrata
                if (currentLevel > hogBossLevelBefore)
                {
                    DesktopLogger.Log($"[Victory] HoG statue level changed: {hogBossLevelBefore} -> {currentLevel}");
                    return true;
                }

                // CORREZIONE: Cerca l'oggetto BossSequenceController nella scena invece di usare FindObjectOfType
                GameObject bscObject = GameObject.Find("BossSequenceController");
                if (bscObject != null)
                {
                    // BossSequenceController gestisce la sequenza nella HoG
                    // Se ha registrato il completamento, siamo a posto
                    var fsm = bscObject.GetComponent<PlayMakerFSM>();
                    if (fsm != null)
                    {
                        string state = fsm.ActiveStateName.ToLower();
                        if (state.Contains("win") || state.Contains("complete") || state.Contains("end"))
                        {
                            return true;
                        }
                    }
                }

                return false;
            }
            catch
            {
                return false;
            }
        }

        private bool CheckBattleControlVictory()
        {
            try
            {
                // Cerca la FSM "Battle Control" o "Battle Scene" nella scena
                PlayMakerFSM[] allFsms = GameObject.FindObjectsOfType<PlayMakerFSM>();
                foreach (var fsm in allFsms)
                {
                    string fsmName = fsm.FsmName.ToLower();
                    if (fsmName.Contains("battle") || fsmName.Contains("boss scene"))
                    {
                        string state = fsm.ActiveStateName.ToLower();
                        if (state.Contains("win") || state.Contains("victory") ||
                            state.Contains("end") || state.Contains("complete") ||
                            state.Contains("defeated"))
                        {
                            DesktopLogger.Log($"[Victory] Battle FSM '{fsm.FsmName}' in victory state: {fsm.ActiveStateName}");
                            return true;
                        }
                    }
                }
                return false;
            }
            catch
            {
                return false;
            }
        }

        private void ForceHoGVictoryRegistration()
        {
            try
            {
                DesktopLogger.Log("[Victory] Attempting forced victory registration...");

                var pd = PlayerData.instance;
                if (pd == null) return;

                // CORREZIONE: Cerca l'oggetto BossSequenceController invece di usare FindObjectOfType
                GameObject bscObject = GameObject.Find("BossSequenceController");
                if (bscObject != null)
                {
                    // Cerca la PlayMakerFSM principale su questo oggetto
                    PlayMakerFSM bscFsm = bscObject.GetComponent<PlayMakerFSM>();
                    if (bscFsm != null)
                    {
                        // Prova a inviare eventi di vittoria
                        try
                        {
                            bscFsm.SendEvent("WIN");
                            bscFsm.SendEvent("BOSS DEAD");
                            bscFsm.SendEvent("END");
                            DesktopLogger.Log("[Victory] ✓ Sent victory events to BossSequenceController FSM");
                        }
                        catch (Exception e)
                        {
                            DesktopLogger.LogWarning($"[Victory] FSM events failed: {e.Message}");
                        }
                    }
                }

                // Fallback: cerca e triggera la FSM di vittoria
                PlayMakerFSM[] allFsms = GameObject.FindObjectsOfType<PlayMakerFSM>();
                foreach (var fsm in allFsms)
                {
                    string fsmName = fsm.FsmName.ToLower();
                    if (fsmName.Contains("battle") || fsmName.Contains("boss scene"))
                    {
                        // Prova a inviare l'evento "WIN" alla FSM
                        try
                        {
                            fsm.SendEvent("WIN");
                            fsm.SendEvent("BOSS DEAD");
                            fsm.SendEvent("BATTLE END");
                            DesktopLogger.Log($"[Victory] Sent victory events to FSM: {fsm.FsmName}");
                        }
                        catch { }
                    }
                }
            }
            catch (Exception e)
            {
                DesktopLogger.LogError($"[Victory] ForceRegistration error: {e.Message}");
            }
        }

        private int GetCurrentBossStatueLevel()
        {
            try
            {
                // In HoG, ogni boss ha una statua con un livello di completamento.
                // La variabile PlayerData è tipicamente nomata:
                // "statueState" + nome del boss, o accessibile via BossStatue component.

                // Cerca l'oggetto BossStatue nella scena
                var statues = GameObject.FindObjectsOfType<BossStatue>();
                foreach (var statue in statues)
                {
                    // BossStatue ha un campo che indica il completamento
                    try
                    {
                        // Accediamo via reflection perché la struttura può variare
                        var completionField = statue.GetType().GetField("bossStatueComplete",
                            System.Reflection.BindingFlags.Public |
                            System.Reflection.BindingFlags.NonPublic |
                            System.Reflection.BindingFlags.Instance);

                        if (completionField != null)
                        {
                            var completion = completionField.GetValue(statue);
                            if (completion != null)
                            {
                                // BossStatue.Completion ha i campi:
                                // isUnlocked, completedTier1, completedTier2, completedTier3
                                var t1 = completion.GetType().GetField("completedTier1");
                                var t2 = completion.GetType().GetField("completedTier2");
                                var t3 = completion.GetType().GetField("completedTier3");

                                int level = 0;
                                if (t1 != null && (bool)t1.GetValue(completion)) level = 1;
                                if (t2 != null && (bool)t2.GetValue(completion)) level = 2;
                                if (t3 != null && (bool)t3.GetValue(completion)) level = 3;

                                return level;
                            }
                        }
                    }
                    catch { }
                }

                return -1; // Non trovato
            }
            catch
            {
                return -1;
            }
        }
    }
}
