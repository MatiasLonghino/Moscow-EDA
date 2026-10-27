const fs = require('fs');
const path = require('path');
const {marked} = require('marked');
const {chromium} = require('playwright');
(async () => {
  const folder = __dirname;
  const parts = JSON.parse(fs.readFileSync(path.join(folder, 'comparacion.json'), 'utf8'));
  const html = '<!doctype html><meta charset="utf-8"><style>body{font:16px Arial;max-width:1150px;margin:32px auto;color:#22372f}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:8px;border-bottom:1px solid #ddd}th{text-align:left}p{line-height:1.5}details{margin-top:24px}summary{cursor:pointer}</style>' + parts.map(p => p.html || marked.parse(p.markdown)).join('\n');
  fs.writeFileSync(path.join(folder,'comparacion.html'),html);
  const browser = await chromium.launch({headless:true,channel:'msedge'});
  const page = await browser.newPage({viewport:{width:1280,height:1000},deviceScaleFactor:1});
  await page.goto('file:///' + path.join(folder,'comparacion.html').replaceAll('\\','/'));
  await page.screenshot({path:path.join(folder,'comparacion.png'),fullPage:true});
  console.log('Vista de comparación generada; anexos cerrados:', await page.locator('details:not([open])').count());
  await browser.close();
})();
