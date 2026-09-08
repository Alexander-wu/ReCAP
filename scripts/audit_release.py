"""Fail CI on private endpoints, credentials, broken JSON or syntax errors."""
import ast,json,re,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
patterns=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'hf_[A-Za-z0-9]{30,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',r'/Users/[^/]+/',r'https?://(?:\d{1,3}\.){3}\d{1,3}',r'/dockerdata/',r'star-proxy\.',r'\b(?:CS2|CrossFPS|crossfps|d2e_ivideogpt)\b']
failures=[];count=0
for p in root.rglob('*'):
    if not p.is_file() or any(x in p.parts for x in ['.git','weights','.venv','__pycache__','build','dist']) or any(x.endswith('.egg-info') for x in p.parts):continue
    if p.suffix not in ('.py','.json','.md','.txt','.yaml','.yml','.toml','.sh','.html','.cff','.csv'):continue
    s=p.read_text();count+=1
    if p.name!='audit_release.py':
        for pattern in patterns:
            if re.search(pattern,s):failures.append(f'{p.relative_to(root)}: private endpoint or credential pattern')
    try:
        if p.suffix=='.py':ast.parse(s)
        if p.suffix=='.json':json.loads(s)
    except (SyntaxError,ValueError) as e:failures.append(f'{p.relative_to(root)}: {e}')
print(json.dumps({'text_files_checked':count,'failures':failures},indent=2))
if failures:sys.exit(1)
