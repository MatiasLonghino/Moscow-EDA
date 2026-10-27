import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const mod='C:/Users/mglon/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const {PresentationFile,FileBlob}=await import(pathToFileURL(mod).href);
const base=path.dirname(fileURLToPath(import.meta.url));
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(base,'output/Propiedades_Usadas_Moscu.pptx')));
await fs.mkdir(path.join(base,'render'),{recursive:true});
for(const [i,s] of p.slides.items.entries()){
 const b=await p.export({slide:s,format:'png',scale:1.4});
 await fs.writeFile(path.join(base,'render',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
 console.log(`Rendered ${i+1}`);
}
