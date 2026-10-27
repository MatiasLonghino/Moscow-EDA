import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const base=path.dirname(fileURLToPath(import.meta.url));
const workspace=path.resolve(base,'../..');
const skill='C:/Users/mglon/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const r=await finalizePresentation({workspaceDir:workspace,candidatePath:path.join(base,'output/Propiedades_Usadas_Moscu.pptx'),finalPath:path.join(workspace,'Presentacion_Propiedades_Usadas/Propiedades_Usadas_Moscu.pptx'),explicitTotalSlideCount:15,pythonExecutable:path.join(workspace,'.venv/Scripts/python.exe'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[6,7,8,9,10,11].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[6,7,8,9,10,11],requiredNativeChartOwnerSlides:[3,4,12],fontPolicy:{basis:'design',families:['Arial','Segoe UI Symbol']},verifyArtifactToolImport:true,receiptPath:path.join(base,'final-validation.json')});
console.log(JSON.stringify(r));


