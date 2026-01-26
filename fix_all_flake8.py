import re
import os

def fix_file(filepath):
    """Corregge automaticamente gli errori Flake8."""

    if not os.path.exists(filepath):
        print(f"✗ File non trovato: {filepath}")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Fix F541: rimuovi f da f-string senza placeholder (senza { })
    content = re.sub(r'print\(f"([^"{}]*?)"\)', r'print("\1")', content)
    content = re.sub(r'f\.write\(f"([^"{}]*?)"\)', r'f.write("\1")', content)

    # Fix E226: aggiungi spazi intorno agli operatori aritmetici
    # episode+1 -> episode + 1
    content = re.sub(r'(\w+)(\+)(\d+)', r'\1 \2 \3', content)
    content = re.sub(r'(\w+)(\-)(\d+)', r'\1 \2 \3', content)
    content = re.sub(r'(\d+)(\+)(\w+)', r'\1 \2 \3', content)
    content = re.sub(r'(\d+)(\-)(\w+)', r'\1 \2 \3', content)

    # Fix E203: rimuovi spazio prima di : nello slicing
    content = re.sub(r'\[(\w+)\s+:\s+', r'[\1:', content)

    # Fix E722: sostituisci bare except con except Exception
    content = re.sub(r'(\s+)except:\s*\n', r'\1except Exception:\n', content)

    # Salva solo se modificato
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed: {filepath}")
        return True
    else:
        print(f"  Unchanged: {filepath}")
        return False

# File da correggere
files_to_fix = [
    "AI_Agents/scripts/train_dqn.py",
    "AI_Agents/scripts/train_ppo.py",
    "AI_Agents/src/agents/ppo_agent.py",
    "AI_Agents/src/env/hollow_knight_env.py",
    "AI_Agents/src/models/dqn_net.py",
    "AI_Agents/src/utils/generate_plots.py",
]

print("=== Correzione automatica errori Flake8 ===\n")
fixed_count = 0

for filepath in files_to_fix:
    try:
        if fix_file(filepath):
            fixed_count += 1
    except Exception as e:
        print(f"✗ Errore su {filepath}: {e}")

print(f"\n{fixed_count} file corretti automaticamente!")
print("\n=== Correzioni manuali richieste ===")
print("Gli errori E402 in train_dqn.py e train_ppo.py:")
print("  Sposta 'import subprocess' SOPRA la riga sys.path.insert(...)")
