"""Build a self-contained, verified handoff snapshot. Does not recompute research."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys,zipfile

BASE=Path(__file__).resolve().parent
NAME='Project3_RMT_handoff_2026-11-10'
PARENT=BASE/'handoff';DEST=PARENT/NAME;ARCHIVE=PARENT/(NAME+'.zip')
EXCLUDED={'.DS_Store','__pycache__','node_modules','.venv','.git'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def copy_file(source,rel):
    if source.name.startswith('~$') or source.name in EXCLUDED:return
    target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)

def copy_tree(source,rel):
    for p in sorted(source.rglob('*')):
        parts=p.relative_to(source).parts
        if any(a in EXCLUDED or a.startswith('~$') or a.startswith('.') for a in parts):continue
        if p.is_file() and not p.is_symlink():copy_file(p,Path(rel)/p.relative_to(source))

def main():
    if DEST.exists() or ARCHIVE.exists():raise FileExistsError('Snapshot already exists; choose a new NAME before regenerating.')
    DEST.mkdir(parents=True)
    copy_file(BASE/'HANDOFF_START_HERE.md','HANDOFF_START_HERE.md')
    for p in sorted((BASE/'handoff_docs').glob('*.md')):copy_file(p,Path('handoff_docs')/p.name)
    copy_file(BASE/'handoff_docs/verify_bundle.py','tools/verify_bundle.py')
    rel=Path('Project3/Analysis_codex')
    for pattern in ['*.py','*.md','*.json','requirements*.txt']:
        for p in sorted(BASE.glob(pattern)):
            if p.name=='HANDOFF_START_HERE.md':continue
            copy_file(p,rel/p.name)
    for folder in ['tables','figures','rmt_tables','rmt_figures','api_refresh','presentation_data',
                   'presentation_output','revision_data','revision_figures','revision_output']:
        copy_tree(BASE/folder,rel/folder)
    for p in sorted((BASE/'presentation_build').glob('*.mjs')):copy_file(p,rel/'presentation_build'/p.name)
    # Rendering and authoring outputs in build folders are intentionally excluded.
    for directory,names in {
        'presentation_build':['validation.json','validation-final.json'],
        'revision_build':['revision_validation.json','preserved_workbooks.json','slide_mapping.json']}.items():
        for name in names:
            p=BASE/directory/name
            if p.exists():copy_file(p,Path('validation_records')/directory/name)
    copy_tree(BASE.parent/'Data/沪深300','Project3/Data/沪深300')
    for name in ['4-金融市场的价格波动.pdf','5-金融商品价格的相关.pdf']:
        copy_file(BASE.parents[1]/'Slides'/name,Path('Slides')/name)
    temp=Path('/var/folders/sw/rsjs7dyx2q3fr5d94pm4cfyh0000gn/T')
    optional={
        'codex-clipboard-28c55b81-2b65-43b5-8c40-0fdce120e1d5.png':'course_project_requirements.png',
        'codex-clipboard-00d411fa-23ab-4a53-9d88-5fb26927bda9.png':'course_assessment_rubric.png'}
    attached=[]
    for source,name in optional.items():
        if (temp/source).exists():copy_file(temp/source,Path('reference_materials')/name);attached.append(name)
    python=BASE/'.venv/bin/python'
    cmd="import sys,platform,importlib.metadata as m; print('Python:',sys.version); print('Platform:',platform.platform()); print('Architecture:',platform.machine()); [print(n+'=='+m.version(n)) for n in ['numpy','pandas','scipy','matplotlib','seaborn','yfinance']]"
    env=subprocess.check_output([str(python),'-c',cmd],text=True)
    (DEST/'handoff_docs/environment_snapshot.txt').write_text(env,encoding='utf-8')
    ppt=BASE/'revision_output/经济物理_Project3_修订版.pptx'
    validation=json.loads((BASE/'revision_build/revision_validation.json').read_text())
    with zipfile.ZipFile(ppt) as z:
        slides=len([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')])
        notes=len([n for n in z.namelist() if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml')])
    snapshot={
      'client_date':'2026-11-10','packaged_at_utc':datetime.now(timezone.utc).isoformat(),
      'latest_pptx':'Project3/Analysis_codex/revision_output/经济物理_Project3_修订版.pptx',
      'slides':slides,'speaker_notes':notes,'current_pptx_sha256':sha(ppt),
      'previous_generated_sha256':validation['finalSha256'],
      'matches_previous_generated_validation':sha(ppt)==validation['finalSha256'],
      'version_notice':'Current disk PPTX is authoritative; rebuilding may discard edits made after code generation.',
      'raw_pickle_count':len(list((BASE.parent/'Data/沪深300').rglob('*.pkl'))),
      'source_code':'Copied byte-for-byte. Absolute runtime paths in presentation JS need local adaptation.',
      'optional_reference_attachments':attached,
      'excluded':['virtual environments','node_modules','Python caches','temporary PowerPoint locks',
                  'draft PPTX files','slide render caches','build logs']}
    (DEST/'handoff_docs/SNAPSHOT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[]
    for p in sorted(DEST.rglob('*')):
        if p.is_file():files.append({'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest={'format':'econophysics-project3-handoff-v1','files':files,
              'file_count_excluding_manifest':len(files),'total_uncompressed_bytes_excluding_manifest':sum(f['bytes'] for f in files)}
    (DEST/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(DEST/'tools/verify_bundle.py')],check=True)
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,Path(NAME)/p.relative_to(DEST))
    with zipfile.ZipFile(ARCHIVE) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('ZIP integrity failure: '+bad)
        # Compare every archived payload directly against its uncompressed snapshot hash.
        for f in files:
            if hashlib.sha256(z.read(NAME+'/'+f['path'])).hexdigest()!=f['sha256']:
                raise RuntimeError('ZIP content mismatch: '+f['path'])
    digest=sha(ARCHIVE)
    (PARENT/(NAME+'.zip.sha256')).write_text(digest+'  '+ARCHIVE.name+'\n',encoding='utf-8')
    print(json.dumps({'archive':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'sha256':digest,
                      'files':len(files)+1,'raw_pickle_count':snapshot['raw_pickle_count'],
                      'slides':slides,'current_ppt_matches_original_validation':snapshot['matches_previous_generated_validation']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
