"""Read-only integrity check for the public publication subset."""
from pathlib import Path
import hashlib,json,os

def verify(root):
    relative='evidence/PUBLIC_FILES_SHA256.json'
    manifest=json.loads((root/relative).read_text(encoding='utf-8'))
    expected={x['path'] for x in manifest['files']}|{relative}
    observed=set()
    for parent,dirs,files in os.walk(root,followlinks=False):
        parent=Path(parent)
        if parent==root and '.git' in dirs:dirs.remove('.git')
        for name in dirs+files:
            if (parent/name).is_symlink():raise ValueError('Links are not allowed')
        observed.update((parent/name).relative_to(root).as_posix() for name in files)
    if expected!=observed:raise ValueError('Public file set differs from manifest')
    for row in manifest['files']:
        path=root/row['path'];data=path.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('File differs: '+row['path'])
    selected=json.loads((root/'evidence/PUBLIC_SELECTION.json').read_text(encoding='utf-8'))
    for row in selected['original_source_files']:
        if hashlib.sha256((root/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Original retained source changed')
    return {'status':'PUBLIC_ARCHIVE_VERIFIED','files':len(observed),'retained_original_source_files':len(selected['original_source_files']),'new_training':False}

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
