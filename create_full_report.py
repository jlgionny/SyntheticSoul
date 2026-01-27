import os

def merge_project_files(output_file="output.txt", source_dir="."):
    """
    Legge tutti i file utili del progetto (C#, Python, Config) 
    e li unisce in un unico file di report.
    """
    
    # 1. ESTENSIONI DA INCLUDERE
    # Qui diciamo allo script quali tipi di file ci interessano
    included_extensions = {
        '.cs',      # Codice C# (Unity Mod)
        '.py',      # Codice Python (AI Agent)
        '.csproj',  # Configurazione Progetto C#
        '.sln',     # Soluzione Visual Studio
        '.txt',     # Requirements.txt
        '.md',      # Readme
        '.yml',     # GitHub Actions
        '.json'     # Eventuali config JSON
    }

    # 2. CARTELLE DA IGNORARE (Blacklist)
    # Se il percorso contiene una di queste parole, viene saltato.
    ignored_dirs = {
        '.git', '.github', '.vs', '.idea', '.vscode', # Config IDE/Git
        'bin', 'obj',                                 # File compilati C# (inutili da leggere)
        'venv', 'env', '__pycache__',                 # Librerie Python e cache
        'logs', 'models', 'docs',                     # Cartelle di output
        'build', 'dist', 'Library', 'Temp'            # Cartelle temporanee Unity
    }

    print(f"--- INIZIO SCANSIONE PROGETTO ---")
    print(f"Cartella: {os.path.abspath(source_dir)}")
    
    file_count = 0
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Intestazione del file finale
        outfile.write("PROJECT SOURCE CODE DUMP\n")
        outfile.write("==================================================\n\n")
        
        for root, dirs, files in os.walk(source_dir):
            # A. FILTRO CARTELLE
            # Modifica la lista 'dirs' in-place per impedire a os.walk di entrare nelle cartelle ignorate
            # Questo è fondamentale per non scansionare le migliaia di file di 'venv' o 'Library'
            dirs[:] = [d for d in dirs if d not in ignored_dirs and d.lower() not in ignored_dirs]

            for file in files:
                # B. FILTRO ESTENSIONI
                _, ext = os.path.splitext(file)
                
                # Se l'estensione è nella nostra lista "buona" E non è il file di output stesso
                if ext.lower() in included_extensions and file != output_file:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, source_dir)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                            content = infile.read()
                            
                            # Scrittura intestazione file
                            outfile.write(f"{'='*80}\n")
                            outfile.write(f"FILE: {rel_path}\n")
                            outfile.write(f"{'='*80}\n\n")
                            
                            # Scrittura contenuto
                            outfile.write(content)
                            outfile.write("\n\n") 
                            
                            print(f"--> Aggiunto: {rel_path}")
                            file_count += 1
                            
                    except Exception as e:
                        print(f"Errore lettura {file_path}: {e}")

    print(f"\n--- COMPLETATO ---")
    print(f"Ho unito {file_count} file nel file: {output_file}")

if __name__ == "__main__":
    merge_project_files()