from pathlib import Path
p=Path(__file__).with_name('build.cjs');s=p.read_text(encoding='utf-8-sig')
s=s.replace("const pptx=new pptxgen();", "const H=JSON.parse(fs.readFileSync(path.join(base,'mlflow/comparison.json')));\nconst pptx=new pptxgen();")
s=s.replace("chartColors:[P.teal,P.red]", "chartColors:[P.teal]")
# Keep labels unique so category positions survive import/render. The native axis controls label frequency.
start=s.index("for(const [key,x,title]")
end=s.index("\n+",start) if '\n+' in s[start:] else -1
lineend=s.index('\n',start)
s=s[:start]+"for(const [key,x,title] of [['price',.55,'Millones de RUB'],['area',4.85,'m²']]){chart(s,pptx.ChartType.bar,[series('Anuncios',C[key])],x,2,4.0,2.65,{barDir:'col',gapSizePct:8,catAxisLabelFrequency:'6',catAxisLabelFontSize:9,showCatName:false,catAxisTitle:title});}"+s[lineend:]
s=s.replace("catAxisOrientation:'maxMin'});", "catAxisOrientation:'maxMin',valAxisMajorUnit:3000});",1)
s=s.replace("let monthly=structuredClone(C.monthly);monthly.labels=monthly.labels.map((l,i)=>i%12===0?l.slice(0,4):'');", "let monthly=structuredClone(C.monthly);")
s=s.replace("valAxisTitle:'Millones de RUB',lineSize:2,catAxisLabelFontSize:10", "valAxisTitle:'Millones de RUB',lineSize:2,catAxisLabelFontSize:8,catAxisLabelFrequency:'12'")
start=s.index('//9\n');end=s.index('//11\n',start)
new='''//9
s=slide('Características seleccionadas por cada modelo','Fuente: selecciones registradas en MLflow · universos iniciales de 177 columnas',`Matriz construida dinámicamente desde params.selected_features de cuatro Runs históricos y resultados.json del Run final ${runURL}. Forward 8, Backward 12, Group Lasso 12 y BSS histórico 12. Unión de 13 características, con 7 comunes a los tres orígenes, 5 presentes en dos y 1 exclusiva. BSS histórico evaluó 8.192 combinaciones incluido el nulo, descartó living_area y coincidió con Backward. La última columna identifica la selección temporal final. Las tablas no incorporan selecciones manuales.`);
const models=['forward','backward','lasso','bss'];
const selections=models.map(m=>new Set(JSON.parse(H[m].params.selected_features)));selections.push(new Set(R.selected_features));
const union=[...new Set(selections.slice(0,3).flatMap(v=>[...v]))];
const count=g=>selections.slice(0,3).filter(v=>v.has(g)).length;
union.sort((a,b)=>count(b)-count(a)||a.localeCompare(b));
const mr=[['Característica','Forward\\n177','Backward\\n177','Group Lasso\\n177','BSS\\nhistórico','BSS final\\ntemporal'],...union.map(g=>[g,...selections.map(a=>a.has(g)?'✓':'—')]),['Total',...selections.map(v=>String(v.size))]];
s.addTable(mr.map((row,i)=>row.map((t,j)=>({text:t,options:{fontFace:j>0&&i>0?'Segoe UI Symbol':'Arial',bold:i===0||i===mr.length-1||j===5,color:i===0?P.white:(j===5?P.teal:P.ink),fill:i===0?P.teal:(j===5?P.pale:'FFFFFF'),align:j===0?'left':'center',border:[{color:(i===8?P.teal:P.line),pt:i===8?1.4:.35},{color:'FFFFFF',pt:0},{color:P.line,pt:.35},{color:'FFFFFF',pt:0}]}}))),{x:.6,y:1.45,w:12.1,colW:[3.65,1.65,1.65,1.8,1.6,1.75],rowH:.30,h:4.48,fontFace:'Arial',fontSize:13.5,margin:[.035,.10,.035,.10],valign:'mid',autoPage:false});
para(s,'Forward fue el más selectivo de los tres métodos originales.\\nBackward y BSS histórico conservaron las mismas 12 características.\\nLa validación temporal redujo la selección final a seis grupos.',.65,6.07,11.9,.78,14.5);
//10
s=slide('Rendimiento histórico y elección temporal','Fuente: Runs históricos y BSS_07 temporal · ambos esquemas se presentan por separado',`Métricas históricas recuperadas de MLflow: validación interna aleatoria 80/20 de Train, semilla 42, con Train filtrado (37.984 filas). La validación participó en la selección, no es Test. Backward y BSS histórico coinciden en subconjunto y RMSE para el criterio y universo considerados. Temporal: Train crudo 39.998 filas, tres folds expansivos. BSS vuelve a seleccionar dentro de cada entrenamiento y el RMSE se calcula en períodos posteriores. Mínimo de media k=12: 2.324.850 RUB. Regla 1-SE: umbral 2.531.404 RUB y k=6: 2.480.192 RUB; k=5 supera el umbral. Se acepta 6,68% más RMSE medio a cambio de mitad de grupos. La regla de parsimonia no garantiza eliminar overfitting ni superioridad en Test. Unión y vocabulario OHE heredados de todo Train, de modo que la validación es condicional a esa preselección. No comparar directamente ambas tablas.`);
label(s,'Recorrido histórico',.65,1.42,6.5);label(s,'Validación temporal: 3 folds',8.25,1.42,4.5);
const names={forward:'Forward — 177',backward:'Backward — 177',lasso:'Group Lasso — 177',bss:'BSS histórico'};
const histRows=[['Modelo','Grupos','Columnas','RMSE interno\\n(RUB)'],...models.map((m,i)=>{let q=H[m];return[names[m],selections[i].size,JSON.parse(q.params.selected_columns||q.params.selected_predictors||'[]').length||({forward:141,backward:145,lasso:145,bss:145}[m]),Math.round(q.metrics.selection_rmse_validation_original).toLocaleString('es-AR')]})];
table(s,histRows,.6,1.98,7.2,[2.6,1.05,1.15,2.4],.58,14.5);
table(s,[['Criterio','Grupos','RMSE medio\\n(RUB)'],['Mínimo',12,Math.round(R.rows.find(r=>r.k===R.rule.k_min).mean).toLocaleString('es-AR')],['1-SE · elegido',6,Math.round(R.rule.selected_mean).toLocaleString('es-AR')]],8.2,1.98,4.5,[1.6,.75,2.15],.76,14.5);
para(s,'BSS histórico igualó a Backward. La búsqueda exhaustiva no mejoró ese subconjunto para el criterio considerado.',.65,5.15,6.75,1.0,18);
para(s,'1-SE admite más error y favorece menos grupos. El modelo de seis no obtuvo el RMSE mínimo.',8.3,5.15,4.25,1.0,18);
text(s,'Evaluaciones distintas: validación interna histórica y validación cronológica sobre Train crudo.',.65,6.46,11.9,.35,15,{color:P.red,bold:true});
'''
s=s[:start]+new+s[end:]
s=s.replace("s=slide('Seis características, 139 columnas transformadas'", "s=slide('Modelo final seleccionado: BSS + validación cruzada temporal'")
s=s.replace("text(s,title,.55,.42,12.1,.75,31,{bold:true})", "text(s,title,.55,.42,12.1,.75,title.length>64?27:31,{bold:true})")
s=s.replace("La parsimonia reduce grupos originales. La codificación de district conserva gran parte de la complejidad.", "Modelo final: 6 características, 139 columnas más intercepto.\\nRMSE temporal medio: 2.480.192 RUB. District aporta 128 columnas.")
s=s.replace(".65,6.16,11.9,.6,21", ".65,6.08,11.9,.72,18")
# Native categorical series use a uniform color. Two-series comparisons are explicit only.
s=s.replace("const final=path.join", "const final=path.join")
s=s.replace("pptx.writeFile({fileName:final});", """(async()=>{await pptx.writeFile({fileName:final});
// Remove orphan content-type declarations emitted by the bundled generator.
const JSZip=require('jszip');const zip=await JSZip.loadAsync(fs.readFileSync(final));let ct=await zip.file('[Content_Types].xml').async('string');
ct=ct.replace(/<Override[^>]*PartName=\"([^\"]+)\"[^>]*\\/>/g,(full,part)=>zip.file(part.replace(/^\\//,''))?full:'');
zip.file('[Content_Types].xml',ct);fs.writeFileSync(final,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));})();""")
p.write_text(s,encoding='utf-8')
