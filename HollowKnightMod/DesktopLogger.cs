using System;
using System.IO;

namespace SyntheticSoulMod
{
    public static class DesktopLogger
    {
        // FIX: Custom log file on Desktop for debugging
        private static readonly string LOG_PATH = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            "SyntheticSoul_Debug.txt"
        );

        private static readonly object lockObj = new object();

        static DesktopLogger()
        {
            // Create or clear log file on initialization
            try
            {
                File.WriteAllText(LOG_PATH, "=== SyntheticSoul Mod Debug Log ===\n");
                File.AppendAllText(LOG_PATH, $"Started: {DateTime.Now}\n");
                File.AppendAllText(LOG_PATH, "============================================================\n\n");
            }
            catch (Exception e)
            {
                // Fallback if Desktop path fails
                Modding.Logger.LogError($"Failed to init desktop log: {e.Message}");
            }
        }

        public static void Log(string message)
        {
            lock (lockObj)
            {
                try
                {
                    string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
                    string logLine = $"[{timestamp}] {message}\n";
                    File.AppendAllText(LOG_PATH, logLine);
                }
                catch
                {
                    // Silent fail - don't crash the mod
                }
            }
        }

        public static void LogError(string message)
        {
            lock (lockObj)
            {
                try
                {
                    string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
                    string logLine = $"[{timestamp}] ERROR: {message}\n";
                    File.AppendAllText(LOG_PATH, logLine);
                }
                catch
                {
                    // Silent fail
                }
            }
        }

        public static void LogWarning(string message)
        {
            lock (lockObj)
            {
                try
                {
                    string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
                    string logLine = $"[{timestamp}] WARNING: {message}\n";
                    File.AppendAllText(LOG_PATH, logLine);
                }
                catch
                {
                    // Silent fail
                }
            }
        }
    }
}