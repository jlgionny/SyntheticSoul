import re
import os

def fix_scientific_notation(filepath):
    """Corregge numeri scientifici rotti tipo '2e - 4' -> '2e-4'."""

    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Fix: 1e - 4 -> 1e-4, 2e - 4 -> 2e-4, ecc.
    content = re.sub(r'(\d+)e\s*-\s*(\d+)', r'\1e-\2', content)
    content = re.sub(r'(\d+)e\s*\+\s*(\d+)', r'\1e+\2', content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed: {filepath}")
        return True
    return False

files = [
    "AI_Agents/src/agents/ppo_agent.py",
    "AI_Agents/scripts/train_dqn.py",
    "AI_Agents/scripts/train_ppo.py",
]

print("Correzione numeri scientifici...\n")
for f in files:
    fix_scientific_notation(f)

print("\nFatto! Ora esegui: black AI_Agents/")
