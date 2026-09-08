"""Create a reproducible source archive; weights are distributed separately."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
excluded={'.git','.venv','__pycache__','.pytest_cache','build','dist'}
files=[p for p in sorted(root.rglob('*')) if p.is_file() and not (set(p.relative_to(root).parts)&excluded) and p.relative_to(root).parts[0] not in {'weights','data','runs'} and not any(x.endswith('.egg-info') for x in p.parts) and p.name not in {'SOURCE_MANIFEST.json','.DS_Store'} and not p.name.startswith('.env')]
records=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
manifest=root/'SOURCE_MANIFEST.json';manifest.write_text(json.dumps({'schema_version':1,'files':records},indent=2)+'\n')
dist=root/'dist';dist.mkdir(exist_ok=True);out=dist/'recap-0.1.0-source.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[manifest]:
        info=zipfile.ZipInfo('recap-0.1.0/'+p.relative_to(root).as_posix(),date_time=(2026,9,7,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes())
(out.with_suffix('.zip.sha256')).write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')
print(out)
