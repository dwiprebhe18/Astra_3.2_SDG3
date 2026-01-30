import shutil
from pathlib import Path

p = Path('child_health_data.csv')
if not p.exists():
    print('child_health_data.csv not found')
    raise SystemExit(1)

backup = p.with_suffix('.csv.bak')
shutil.copy(p, backup)
print(f'Backup created at {backup}')

with p.open('r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned = []
header = None
for raw in lines:
    line = raw.rstrip('\n')
    s = line.strip()
    if s.startswith('<<<<<<<') or s.startswith('=======') or s.startswith('>>>>>>>'):
        continue
    if 'Child_ID' in line and header is None:
        header = line
        cleaned.append(line + '\n')
        continue
    if 'Child_ID' in line and header is not None:
        # duplicate header, skip
        continue
    # skip empty lines
    if line == '':
        continue
    cleaned.append(line + '\n')

with p.open('w', encoding='utf-8') as f:
    f.writelines(cleaned)

print(f'Cleaned file written. Lines before: {len(lines)}, after: {len(cleaned)}')
