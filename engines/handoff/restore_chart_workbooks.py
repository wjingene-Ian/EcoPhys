"""Retain original embedded chart workbooks after an Artifact Tool import/export.
Copies the actual source bytes, preserving formula provenance; no data fabrication.
"""
from pathlib import Path,PurePosixPath
import json,posixpath,zipfile,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
S=ROOT/'presentation_output/经济物理_Project3_RMT与投资组合.pptx'
C=ROOT/'revision_build/candidate_revision.pptx'
O=ROOT/'revision_build/candidate_with_source_workbooks.pptx'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
def target(z,owner,typ):
    rel=posixpath.join(posixpath.dirname(owner),'_rels',posixpath.basename(owner)+'.rels')
    e=ET.fromstring(z.read(rel))
    for x in e:
        if x.attrib['Type'].endswith('/'+typ):
            return posixpath.normpath(posixpath.join(posixpath.dirname(owner),x.attrib['Target'])).lstrip('/')
    raise ValueError((owner,typ))

mapping=json.loads((ROOT/'revision_build/slide_mapping.json').read_text())
payload={}
with zipfile.ZipFile(S) as source,zipfile.ZipFile(C) as candidate:
    data={n:candidate.read(n) for n in candidate.namelist()}
    for old in [8,10,16,17,22,29]:
        new=next(m['page'] for m in mapping if m['original']==old)
        a=target(source,f'ppt/slides/slide{old}.xml','chart')
        b=target(candidate,f'ppt/slides/slide{new}.xml','chart')
        workbook=target(source,a,'package')
        preserved=f'ppt/embeddings/preserved-source-chart-{old}.xlsx'
        data[preserved]=source.read(workbook)
        # Original charts were not edited; retain original numeric cache and formulas.
        data[b]=source.read(a)
        rel=ET.Element('Relationships',xmlns=NS['p'])
        original=ET.fromstring(source.read(a)).find('a:externalData',NS)
        relid=original.attrib['{'+NS['r']+'}id']
        ET.SubElement(rel,'Relationship',Id=relid,Type=NS['r']+'/package',Target=posixpath.relpath(preserved,posixpath.dirname(b)))
        relpart=posixpath.join(posixpath.dirname(b),'_rels',posixpath.basename(b)+'.rels')
        data[relpart]=ET.tostring(rel,encoding='utf-8',xml_declaration=True)
        payload[str(old)]={'new_slide':new,'source_workbook':workbook,'preserved_workbook':preserved}
    ct=ET.fromstring(data['[Content_Types].xml']);uri='http://schemas.openxmlformats.org/package/2006/content-types'
    ET.register_namespace('',uri)
    if not any(e.attrib.get('Extension')=='xlsx' for e in ct):
        ET.SubElement(ct,'{'+uri+'}Default',Extension='xlsx',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    data['[Content_Types].xml']=ET.tostring(ct,encoding='utf-8',xml_declaration=True)
with zipfile.ZipFile(O,'w',zipfile.ZIP_DEFLATED) as out:
    for name,blob in data.items():out.writestr(name,blob)
(ROOT/'revision_build/preserved_workbooks.json').write_text(json.dumps(payload,indent=2))
print('Preserved six original chart workbooks:',O)
