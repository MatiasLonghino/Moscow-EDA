"""Preserve zero histogram counts omitted by the bundled PPTX writer in XLSX data."""
from pathlib import Path
import zipfile,io,posixpath,re
from xml.dom import minidom
p=Path(__file__).parent/'output/Propiedades_Usadas_Moscu.pptx'
with zipfile.ZipFile(p) as z: parts={n:z.read(n) for n in z.namelist()}
fixed=0
for chart in [n for n in parts if re.fullmatch(r'ppt/charts/chart\d+\.xml',n)]:
    doc=minidom.parseString(parts[chart]); relpath=posixpath.dirname(chart)+'/_rels/'+posixpath.basename(chart)+'.rels'
    rd=minidom.parseString(parts[relpath]);target=next(e.getAttribute('Target') for e in rd.getElementsByTagName('Relationship') if 'package' in e.getAttribute('Type'))
    wp=posixpath.normpath(posixpath.join(posixpath.dirname(chart),target))
    with zipfile.ZipFile(io.BytesIO(parts[wp])) as wz:wparts={n:wz.read(n) for n in wz.namelist()}
    sheet=minidom.parseString(wparts['xl/worksheets/sheet1.xml']); cells={c.getAttribute('r'):c for c in sheet.getElementsByTagName('c')}
    rows={r.getAttribute('r'):r for r in sheet.getElementsByTagName('row')}
    for ref in doc.getElementsByTagName('c:numRef'):
        formula=ref.getElementsByTagName('c:f')[0].firstChild.data
        m=re.fullmatch(r'Sheet1!\$([A-Z]+)\$(\d+):\$[A-Z]+\$\d+',formula)
        if not m:continue
        col,start=m.group(1),int(m.group(2))
        for pt in ref.getElementsByTagName('c:pt'):
            val=pt.getElementsByTagName('c:v')[0].firstChild.data
            cellref=col+str(start+int(pt.getAttribute('idx')))
            c=cells.get(cellref); v=c.getElementsByTagName('v') if c else []
            if v and v[0].firstChild:continue
            assert float(val)==0,(chart,cellref,val)
            if c is None:
                c=sheet.createElement('c');c.setAttribute('r',cellref);rows[str(start+int(pt.getAttribute('idx')))].appendChild(c)
            if v:ve=v[0]
            else:ve=sheet.createElement('v');c.appendChild(ve)
            if c.hasAttribute('t'):c.removeAttribute('t')
            ve.appendChild(sheet.createTextNode('0'));fixed+=1
    wparts['xl/worksheets/sheet1.xml']=sheet.toxml(encoding='UTF-8')
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as wz:
        for n,b in wparts.items():wz.writestr(n,b)
    parts[wp]=buf.getvalue()
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in parts.items():z.writestr(n,b)
print('Restored zero histogram counts in embedded workbooks:',fixed)
